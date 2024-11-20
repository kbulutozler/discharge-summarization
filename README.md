# discharge-summarization
Task: summarizing discharge reports from mimic iv with llms and pretrained lms


original data from https://github.com/AI-in-Health/ClinicBench

1) zero shot with custom prompt comparison to clinicbench's zero shot results
2) finetuning, zero/few shot experiments for custom split of the original data (train:92 dev: 40 test: 250)
3) scoring script from discarge-me shared task: https://github.com/Stanford-AIMI/discharge-me/tree/main/scoring
4) 