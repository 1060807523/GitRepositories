# coding: utf-8
"""
    this programe is to clear driverlog below this dir
    __author__:the_new_one
"""
import os, traceback


# 查找文件名中包含关键词的文件
def search_dir(s, path=os.path.abspath('.'), files=[]):
    try:
        print("开始")
        for x in os.listdir(path):
            path_now = os.path.join(path, x)
            if os.path.isfile(path_now) and s in os.path.splitext(x)[0]:
                print(path_now)
                # 删除查找到的文件
                os.remove(path_now)
                if x not in files:
                    files.append(x)
            elif os.path.isdir(path_now):
                search_dir(s=s, path=os.path.join(path_now), files=files)
        return files
    except Exception as e:
        print(traceback.format_exc())
        print(e)


if __name__ == "__main__":
    print(os.path.isdir("D:\\testDelete\\sss"))
    result = search_dir(s='4)', path="D:\\testDelete")
