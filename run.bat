@echo off
set global_epoch=100
set join_ratio=0.4

set eval_train=1
set eval_test=1
set test_gap=1

set local_epoch=1
set local_lr=0.0001
set momentum=0.5
set weight_decay=0.01
set batch_size=64

set model="avgcnn"



cd /d ".\src\server"
@REM set algorithm=fedavg
@REM set local_lr=0.0002
@REM call:start_experiment

set algorithm=streamingfedavg
set global_epoch=500
set local_lr=0.0001
call:start_experiment
goto:eof

@REM  call this to start  a experiemnt

:start_experiment
@echo on
python %algorithm%.py ^
--global_epoch %global_epoch% ^
--eval_train %eval_train% ^
--eval_test %eval_test%  ^
--join_ratio %join_ratio% ^
--test_gap %test_gap% ^
--local_epoch %local_epoch% ^
--model %model% ^
--momentum %momentum% ^
--weight_decay %weight_decay% ^
--batch_size %batch_size% ^
--local_lr %local_lr%
@echo off
set srcpath=".\out\%algorithm%"
set destpath=".\out\%algorithm%_%date:~0,4%_%date:~5,2%_%date:~8,2%_%time:~0,2%_%time:~3,2%_%time:~6,2%"
xcopy %srcpath% %destpath% /E /I /Y
goto:eof


