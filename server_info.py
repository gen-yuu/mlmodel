
'''
 ['13th corei5 - RTX3060 Ti' '13th corei7 - GTX1080'
 '13th corei5 - GTX1650' '1th Xeon Gold - RTX4070' '13th corei7 - RTX3050'
 '13th corei5 - GTX1080' '13th corei5 - RTX4070'
 '13th corei7 - RTX3060 Ti' '13th corei5 - RTX3050'
 '1th Xeon Gold - GTX1080' '9th corei7 - RTX2080 Ti'
 '13th corei7 - RTX4070']
'''

server_list = ['13th corei5 - RTX3060 Ti', '13th corei7 - GTX1080',
               '13th corei5 - GTX1650', '1th Xeon Gold - RTX4070', '13th corei7 - RTX3050',
               '13th corei5 - GTX1080', '13th corei5 - RTX4070',
               '13th corei7 - RTX3060 Ti', '13th corei5 - RTX3050',
               '1th Xeon Gold - GTX1080', '9th corei7 - RTX2080 Ti',
               '13th corei7 - RTX4070']
cpu_info = {"13th corei5": {"cpu_core": 14,
                            "cpu_boost_clock(GHz)": 5.1, "cpu_thread": 20, "cpu_cache(MB)": 20},
            "13th corei7": {"cpu_core": 16,
                            "cpu_boost_clock(GHz)": 5.2, "cpu_thread": 24, "cpu_cache(MB)": 30},
            "9th corei7": {"cpu_core": 8,
                           "cpu_boost_clock(GHz)": 4.7, "cpu_thread": 8, "cpu_cache(MB)": 12},
            "1th Xeon Gold": {"cpu_core": 4,
                              "cpu_boost_clock(GHz)": 3.7, "cpu_thread": 8, "cpu_cache(MB)": 16.5}}
gpu_info = {"RTX4070": {}}
for server in server_list:
    target = '-'
    idx = server.find(target)
    cpu = server[:idx-1]
    gpu = server[idx+2:]
    print(cpu)
    print(gpu)
