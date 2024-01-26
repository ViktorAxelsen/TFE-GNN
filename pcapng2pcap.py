import os


if __name__ == '__main__':
    ROOT = '/data1/zhz/ISCX-VPN-NonVPN-2016/NonVPN'
    file_list = os.listdir(ROOT)
    for file in file_list:
        if not file.endswith('.pcapng'):
            continue
        file_path = ROOT + '/' + file
        os.system('editcap -F libpcap {} {}'.format(file_path, file_path[:-2]))
        os.system('rm {}'.format(file_path))