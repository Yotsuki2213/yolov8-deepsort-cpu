# 测试安装是否成功
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    print("DeepSORT库安装成功！")
except ImportError as e:
    print(f"安装失败: {e}")
