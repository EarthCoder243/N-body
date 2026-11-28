/*
Crash safe checkpoint I/O. Checkpoint is either complete or ignored, not half read. 
File only counts as complete if its header, size, and end marker all check out
*/

#include "checkpoint.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <filesystem>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define FSYNC_FILE(f) _commit(_fileno(f))
#else
#include <fcntl.h>
#include <unistd.h>
#define FSYNC_FILE(f) fsync(fileno(f))
#endif

struct CheckpointHeader
{
    char magic[4];
    uint32_t version;
    uint64_t n;
    uint64_t step;
    uint64_t seed;
    float dt;
    float eps2; 
    float G;
    uint32_t reserved;
};

static_assert(sizeof(CheckpointHeader) == 48, "Checkpoint Header has unexpected padding");

const char CHECKPOINT_MAGIC[4] = {'N', 'B', 'C', 'K'};
const char END_MARKER[4] = {'D', 'O', 'N', 'E'};
const uint32_t CHECKPOINT_VERSION = 1;

static uint64_t expectedFileSize(uint64_t n)
{
    return sizeof(CheckpointHeader) + 2 * n * sizeof(Vec4) + sizeof(END_MARKER);
}

static bool writeStateFile(const std::string& path, const SimState& state, bool sync)
{
    FILE* f = fopen(path.c_str(), "wb");
    if(f == NULL)
    {
        fprintf(stderr, "Unable to open %s for writing\n", path.c_str());
        return false;
    }

    CheckpointHeader header = {};
    memcpy(header.magic, CHECKPOINT_MAGIC, 4);
    header.version = CHECKPOINT_VERSION;
    header.n = state.pos.size();
    header.step = state.step;
    header.seed = state.seed;
    header.dt = state.params.dt;
    header.eps2 = state.params.eps2;
    header.G = state.params.G;

    size_t n = state.pos.size();
    bool ok = fwrite(&header, sizeof(header), 1, f) == 1;
    ok = ok && fwrite(state.pos.data(), sizeof(Vec4), n, f) == n;
    ok = ok && fwrite(state.vel.data(), sizeof(Vec4), n, f) == n;
    ok = ok && fwrite(END_MARKER, 1, 4, f) == 4; // Marks file as complete
    ok = ok && fflush(f) == 0;
    if(ok && sync)
    {
        ok = FSYNC_FILE(f) == 0; //Data on disk before rename makes it visible
    }
    if(fclose(f) != 0)
    {
        ok = false;
    }
    if(!ok)
    {
        fprintf(stderr, "Failed writing %s\n", path.c_str());
    }
    return ok;
}

static bool readValidHeader(FILE* f, const std::string& path, CheckpointHeader& header)
{
    if(fread(&header, sizeof(header), 1, f) != 1)
    {
        return false;
    }
    if(memcmp(header.magic, CHECKPOINT_MAGIC, 4) != 0 || header.version != CHECKPOINT_VERSION)
    {
        return false;
    }
    std::error_code ec;
    uint64_t size = std::filesystem::file_size(path, ec);
    return !ec && size == expectedFileSize(header.n);
}

static bool isComplete(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");
    if(f == NULL)
    {
        return false;
    }
    CheckpointHeader header;
    char end[4];
    bool ok = readValidHeader(f, path, header);
    ok = ok && fseek(f, -4, SEEK_END) == 0;
    ok = ok && fread(end, 1, 4, f) == 4 && memcmp(end, END_MARKER, 4) == 0;
    fclose(f);
    return ok;
}

static std::vector<uint64_t> listCheckpointSteps(const std::string& dir)
{
    std::vector<uint64_t> steps;
    std::error_code ec;
    std::filesystem::directory_iterator it(dir, ec);
    std::filesystem::directory_iterator endIt;
    for(; !ec && it != endIt; it.increment(ec))
    {
        std::string name = it->path().filename().string();
        if(name.size() != 21 || name.compare(0, 5, "ckpt-") != 0 || name.compare(17, 4, ".bin") != 0)
        {
            continue;
        }
        std::string digits = name.substr(5, 12);
        bool allDigits = true;
        for(int i = 0; i < (int)digits.size(); i++)
        {
            if(!isdigit((unsigned char)digits[i]))
            {
                allDigits = false;
                break;
            }
        }
        if(allDigits)
        {
            steps.push_back(std::stoull(digits));
        }
    }
    std::sort(steps.begin(), steps.end());
    std::reverse(steps.begin(), steps.end());
    return steps;
}

bool saveState(const std::string& path, const SimState& state)
{
    return writeStateFile(path, state, false);
}

bool loadState(const std::string& path, SimState& state)
{
    FILE* f = fopen(path.c_str(), "rb");
    if(f == NULL)
    {
        fprintf(stderr, "Unable to open %s\n", path.c_str());
        return false;
    }
    CheckpointHeader header;
    if(!readValidHeader(f, path, header))
    {
        fprintf(stderr, "%s is not a complete version %u checkpoint\n", path.c_str(), CHECKPOINT_VERSION);
        fclose(f);
        return false;
    }

    state.resize((int)header.n);
    state.step = header.step;
    state.seed = header.seed;
    state.params.dt = header.dt;
    state.params.eps2 = header.eps2;
    state.params.G = header.G;

    char end[4];
    size_t n = (size_t)header.n;
    bool ok = fread(state.pos.data(), sizeof(Vec4), n, f) == n;
    ok = ok && fread(state.vel.data(), sizeof(Vec4), n, f) == n;
    ok = ok && fread(end, 1, 4, f) == 4 && memcmp(end, END_MARKER, 4) == 0;
    fclose(f);
    if(!ok)
    {
        fprintf(stderr, "Failed reading %s\n", path.c_str());
    }
    return ok;
}

std::string checkpointPath(const std::string& dir, uint64_t step)
{
    char name[32];
    snprintf(name, sizeof(name), "ckpt-%012llu.bin", (unsigned long long)step);
    return (std::filesystem::path(dir) / name).string();
}

bool writeCheckpoint(const std::string& dir, const SimState& state, int keep)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string finalPath = checkpointPath(dir, state.step);
    std::string tmpPath = finalPath + ".tmp";

    if(!writeStateFile(tmpPath, state, true))
    {
        return false;
    }
    std::filesystem::rename(tmpPath, finalPath, ec);
    if(ec)
    {
        fprintf(stderr, "Unable to rename %s: %s\n", tmpPath.c_str(), ec.message().c_str());
        return false;
    }
#ifndef _WIN32
    int dirFd = open(dir.c_str(), O_RDONLY);
    if(dirFd >= 0)
    {
        fsync(dirFd);
        close(dirFd);
    }
#endif

    std::vector<uint64_t> steps = listCheckpointSteps(dir);
    for(int i = keep; i < (int)steps.size(); i++)
    {
        std::filesystem::remove(checkpointPath(dir, steps[i]), ec);
    }
    return true;
}

bool newestCheckpoint(const std::string& dir, uint64_t& step)
{
    std::vector<uint64_t> steps = listCheckpointSteps(dir);
    for(int i = 0; i < (int)steps.size(); i++)
    {
        if(isComplete(checkpointPath(dir, steps[i])))
        {
            step = steps[i];
            return true;
        }
    }
    return false;
}

void discardCheckpointsAfter(const std::string& dir, uint64_t step)
{
    std::vector<uint64_t> steps = listCheckpointSteps(dir);
    std::error_code ec;
    for(int i = 0; i < (int)steps.size(); i++)
    {
        if(steps[i] > step)
        {
            std::filesystem::remove(checkpointPath(dir, steps[i]), ec);
        }
    }
}
