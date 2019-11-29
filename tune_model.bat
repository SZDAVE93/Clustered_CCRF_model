@echo off
D:
cd D:\yifei\Documents\Codes_on_GitHub\Clustered_CCRF_model
for %%I in (400,430,460,490,520,550,580,610,640,670,700,730,760,790,820,850,880,910,940,970) do python main_model.py --start_date %%I --eval_days 7 --simi_len 90
for %%I in (400,430,460,490,520,550,580,610,640,670,700,730,760,790,820,850,880,910,940,970) do python main_model.py --start_date %%I --eval_days 14 --simi_len 90
for %%I in (400,430,460,490,520,550,580,610,640,670,700,730,760,790,820,850,880,910,940,970) do python main_model.py --start_date %%I --eval_days 21 --simi_len 90
pause
