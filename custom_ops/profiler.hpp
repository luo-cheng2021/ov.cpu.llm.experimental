#pragma once

#include <map>
#include <deque>
#include <thread>
#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include "x86intrin.h"

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <fcntl.h>           /* For O_* constants */
#include <sys/stat.h>        /* For mode constants */
#include <semaphore.h>

struct ProfileData {
    uint64_t start;
    uint64_t end;
    std::string name;  // Title
    std::string cat;   // Category
    std::map<const char *, std::string> args;

    ProfileData(const std::string& name) : name(name) {
        start = __rdtsc();
    }
};


struct chromeTrace {
    std::ostream& os;
    int fake_tid;
    uint64_t ts;
    chromeTrace(std::ostream& os, int fake_tid) : os(os), fake_tid(fake_tid) {}
    void setTs(uint64_t _ts) {
        ts = _ts;
    }
    void addCounter(std::string name, std::vector<std::pair<std::string, double>> values) {
        // name += std::to_string(fake_tid);
        os << "{\n"
           << "\"ph\": \"C\",\n"
           << "\"name\": \"" << name << "\",\n"
           << "\"pid\": " << fake_tid << ",\n"
           << "\"tid\": " << 0 << ",\n"
           << "\"ts\": " << ts << ",\n"
           << "\"args\": {\n";
        const char* sep = "";
        for (auto& pair : values) {
            os << sep << "\"" << pair.first << "\" : " << pair.second;
            sep = ",";
        }
        os << " }},\n";
    }
    void addCompleteEvent(std::string name,
                          std::string cat,
                          uint64_t start,
                          uint64_t dur,
                          const std::map<const char*, std::string>& args) {
        os << "{\n";
        os << "\"ph\": \"X\",\n"
           << "\"cat\": \"" << cat << "\",\n"
           << "\"name\": \"" << name << "\",\n"
           << "\"pid\": " << fake_tid << ",\n"
           << "\"tid\": " << 0 << ",\n"
           << "\"ts\": " << start << ",\n"
           << "\"dur\": " << dur << ",\n"
           << "\"args\": {\n";
        const char* sep = "";
        for (auto& a : args) {
            std::string key = a.first;
            os << sep << "      \"" << a.first << "\" : \"" << a.second << "\"";
            sep = ",\n";
        }
        os << "\n          }\n";
        os << "},\n";
    }
};


class ProfilerManager {
    bool enabled;
    // cannot use vector<> since each new Profile() API call will
    // emplace_back() an item and wrap it into a shared_ptr, this
    // process is nested and during which vector resize may invalid
    // the ProfileData elements still referenced by an alive shared_ptr
    // and later when it finally gets un-referenced, a wild pointer would
    // be updated and memory would be corrupted. deque can fix it.
    std::deque<ProfileData> all_data;
    uint64_t  tsc_ticks_per_second;
    uint64_t  tsc_ticks_base;
    const char* dump_file_name;
    pid_t thread_id;

public:
    static uint64_t rdtsc_calibrate(int seconds = 1) {
        uint64_t start_ticks;
        start_ticks = __rdtsc();
        std::this_thread::sleep_for(std::chrono::seconds(seconds));
        return (__rdtsc() - start_ticks) / seconds;
    }

    ProfilerManager(const char * dump_file_name = "profile.json") : dump_file_name(dump_file_name) {
        const char* str_enable = std::getenv("LLM_PROFILE");
        if (!str_enable)
            str_enable = "0";
        int num_hint = atoi(str_enable);
        set_enable(num_hint > 0);
        if (enabled) {
            //tsc_ticks_per_second = 3696445548;
            if (tsc_ticks_per_second == 0) {
                auto tps = rdtsc_calibrate();
                tsc_ticks_per_second = tps;
                std::cout << "=== ProfilerManager2: tsc_ticks_per_second = " << tsc_ticks_per_second << std::endl;
                tsc_ticks_base = 0;//__rdtsc();
            }
            thread_id = syscall(__NR_gettid);
        }
    }

    ~ProfilerManager(){
        finalize();
    }

    struct SemMutex {
        sem_t * sem;
        SemMutex() {
            sem = sem_open("/profiler_sem", O_CREAT, 0644, 1);
            if (sem == SEM_FAILED) {
                perror("Failed to open semphore for empty");
                return;
            }
            sem_wait(sem);
        }
        ~SemMutex() {
            if (sem != SEM_FAILED)
                sem_post(sem);
        }
    };

    void SafeDump(const char * dump_file_name, const char * content) {
        SemMutex sem_mutex;

        // the last ProfilerManagers is responsible for dump to file
        auto * fw = fopen(dump_file_name, "r+");    // Open for reading and writing
        if (!fw)
            fw = fopen(dump_file_name, "w");        // Create if not exist

        if (!fw) {
            std::cout << "==== ProfilerManager2 failed to dump into " << dump_file_name << "\n";
            return;
        }

        fseek(fw, 0, SEEK_END);
        auto off0 = ftell(fw);
        if (off0 == 0) {
            // empty file, add header
            fprintf(fw, "{\n\"schemaVersion\": 1,\n\"traceEvents\": [\n");
            fflush(fw);
        } else {
            fseek(fw, -3, SEEK_END); // remove last 3 chars `]` `}` `\n`
            fprintf(fw, ",");
            fflush(fw);
        }

        fprintf(fw, "%s", content);

        fprintf(fw, R"({
                    "name": "Profiler End",
                    "ph": "i",
                    "s": "g",
                    "pid": "Traces",
                    "tid": "Trace OV Profiler",
                    "ts":)");
        fprintf(fw, "%lu}\n", tsc_to_usec(__rdtsc()));
        fprintf(fw, "]}\n");
        fclose(fw);
        std::cout << "==== ProfilerManager2 data is dumpped into " << dump_file_name << " off0=" << off0 << "  \n";
    }

    void finalize() {
        // collect all entries
        std::ostringstream dump_text;
        int64_t total_traces = 0;
        if (all_data.size()) {
            chromeTrace ct(dump_text, thread_id);
            for (auto& d : all_data) {
                ct.addCompleteEvent(d.name, d.cat, tsc_to_usec(d.start), tsc_to_usec(d.end) - tsc_to_usec(d.start), d.args);
                total_traces++;
            }
            std::cout << "==== Profile: total number of profile entries " << all_data.size() << std::endl;
        }

        if (total_traces == 0)
            return;

        SafeDump(dump_file_name, dump_text.str().c_str());
    }

    ProfileData* startProfile(const std::string& name) {
        all_data.emplace_back(name);
        return &all_data.back();
    }
    uint64_t tsc_to_usec(uint64_t tsc_ticks) {
        return (tsc_ticks - tsc_ticks_base) * 1000000 / tsc_ticks_per_second;
    }

    void set_enable(bool on) {
        enabled = on;
    }
    bool is_enabled() {
        return enabled;
    }

    struct ProfileDataWrapper {
        ProfileData * p;
        ProfileDataWrapper(ProfileData * p = nullptr) : p(p) {
        }
        ~ProfileDataWrapper() {
            if (p)
                p->end = __rdtsc();
        }
        ProfileDataWrapper(const ProfileDataWrapper &) = delete;
        ProfileDataWrapper(ProfileDataWrapper && r) {
            p = r.p;
            r.p = nullptr;
        }
        ProfileDataWrapper& operator=(ProfileDataWrapper&& r) {
            p = r.p;
            r.p = nullptr;
            return *this;
        }
    };

    inline ProfileDataWrapper Profile(const char* name) {
        if (!is_enabled())
            return ProfileDataWrapper();
        ProfileData* p = startProfile(name);
        return ProfileDataWrapper(p);
    }

    inline ProfileDataWrapper Profile(const std::string& name) {
        if (!is_enabled())
            return nullptr;
        ProfileData* p = startProfile(name);
        return ProfileDataWrapper(p);
    }

    template<typename ... Ts>
    inline ProfileDataWrapper Profile(Ts... args) {
        if (!is_enabled())
            return ProfileDataWrapper();

        std::stringstream ss;
        int dummy[sizeof...(Ts)] = { (ss << args << " ", 0)... };
        (void)(dummy);
        ProfileData* p = startProfile(ss.str());
        return ProfileDataWrapper(p);
    }
};
