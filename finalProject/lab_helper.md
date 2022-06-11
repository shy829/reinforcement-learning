可以在后台执行程序，比如

`CUDA_VISIBLE_DEVICES=3 nohup python run.py --env_name Humanoid-v2 > log.txt 2>&1 &`

通过查看log文件或者观察 `ps -aux |grep (pid)` 来观察程序是否正常运行