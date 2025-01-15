# server_list = [
#     '13th corei5 - RTX3060 Ti', '13th corei7 - GTX1080', '13th corei5 - GTX1650',
#     '1th Xeon Gold - RTX4070', '13th corei7 - RTX3050', '13th corei5 - GTX1080',
#     '13th corei5 - RTX4070', '13th corei7 - RTX3060 Ti', '13th corei5 - RTX3050',
#     '1th Xeon Gold - GTX1080', '9th corei7 - RTX2080 Ti', '13th corei7 - RTX4070'
# ]
'''
0:NVIDIA Ada Lovelace
1:NVIDIA Ampere
2:NVIDIA Turing
3:NVIDIA Pascal
'''
cpu_info = {
    "13th corei5": {
        "cpu_core": 14,
        "cpu_boost_clock(GHz)": 5.1,
        "cpu_thread": 20,
        "cpu_cache(MB)": 20
    },
    "13th corei7": {
        "cpu_core": 16,
        "cpu_boost_clock(GHz)": 5.2,
        "cpu_thread": 24,
        "cpu_cache(MB)": 30
    },
    "9th corei7": {
        "cpu_core": 8,
        "cpu_boost_clock(GHz)": 4.7,
        "cpu_thread": 8,
        "cpu_cache(MB)": 12
    },
    "1th Xeon Gold": {
        "cpu_core": 4,
        "cpu_boost_clock(GHz)": 3.7,
        "cpu_thread": 8,
        "cpu_cache(MB)": 16.5
    }
}

gpu_info = {
    "RTX4070": {
        "gpu_architecture": 0,
        "gpu_core": 5888,
        "gpu_boost_clock(GHz)": 2.48,
        "VRAM(GB)": 12,
    },
    "RTX3060 Ti": {
        "gpu_architecture": 1,
        "gpu_core": 4864,
        "gpu_boost_clock(GHz)": 1.665,
        "VRAM(GB)": 8,
    },
    "RTX3050": {
        "gpu_architecture": 1,
        "gpu_core": 2560,
        "gpu_boost_clock(GHz)": 1.777,
        "VRAM(GB)": 8,
    },
    "RTX2080 Ti": {
        "gpu_architecture": 2,
        "gpu_core": 4352,
        "gpu_boost_clock(GHz)": 1.640,
        "VRAM(GB)": 11,
    },
    "GTX1650": {
        "gpu_architecture": 2,
        "gpu_core": 896,
        "gpu_boost_clock(GHz)": 1.665,
        "VRAM(GB)": 4,
    },
    "GTX1080": {
        "gpu_architecture": 3,
        "gpu_core": 2560,
        "gpu_boost_clock(GHz)": 1.733,
        "VRAM(GB)": 8,
    }
}


# server : "cpu - gpu"
def get_server_spec(server):
    target = '-'
    idx = server.find(target)
    cpu = server[:idx - 1]
    gpu = server[idx + 2:]
    feature: dict = {}
    feature.update(**cpu_info[cpu], **gpu_info[gpu])
    return feature
