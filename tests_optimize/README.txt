文件夹说明：
    data/       ：官方驱动验证用的数据
    scripts/    ：官方驱动调用的脚本
*   weight/     ：网络指令和参数文件
*   picture/    ：输入列表文件和一些输入图片
*   output/     ：输出结果文件和输出图片

文件说明：
    dma_from_device.c   ：官方驱动调用的程序
    dma_to_device.c     ：官方驱动调用的程序
*   face_dpu_driver.c   ：配置dpu的主要驱动程序
    load_driver.sh      ：官方驱动调用的脚本
*   Makefile            ：编译文件
    performance.c       ：官方驱动调用的程序
    reg_rw.c            ：官方驱动调用的程序
    run_test.sh         ：官方驱动调用的脚本
*   test.py             ：调用dpu实现人脸识别任务的主要脚本

使用说明：
    0. 准备官方驱动文件夹
    1. 将tests_optimize/文件夹拷贝到官方驱动文件夹下，与tests/平级
    2. 按照正常方法编译、安装驱动
    3. 进入tests_optimize/文件夹，执行 make
    4. 在tests_optimize/文件夹下执行 sudo bash ./load_driver.sh 与 ./run_test.sh 确认驱动正常
    5. 在tests_optimize/文件夹下执行 python test.py ./picuture/input_list.txt ./output/results.out
    6. 可以通过修改input_list.txt指定处理的图片对
    7. 可以通过更改results.out的路径更改输出路径
    8. 可以在执行test.py程序时，加上--save_image保存结果对比图，再加上--show在执行中显示图片
    9. 可以在执行test.py程序时，加上--threshold 0.71 修改默认的阈值为0.71
