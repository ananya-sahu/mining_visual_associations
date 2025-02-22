# mining_visual_associations

# Data Generation Process: 

Step 1: format file to get associations from gpt4-o-mini and upload generated files to open ai batch api 

python get_associations_format.py <path to coco train captions> <path to coco val captions> <path to coco train annotations> <path to coco images directory> <path to concreteness ratings file> <path to dense descriptions> <save path for train format file> <save path for val format file> 

Step 2: after getting associations from gpt-4-o-mini in Step 1, format prompts for generating creative captions in Step 3

python format_associations.py <path to train files obtained from step 1 after batch api> <path to val files obtained from step 1 after batch api> <path to coco train captions> <path to coco val captions> <path to where formatted prompts should get saved>


Step 3: after obtaining formatted prompts from step 2, generate creative captions
 
python generate_creative_captions.py <raw prompts file from Step 2> <coco split> <distance label> <environment cache path> <directory path to save outputs>


# Model Training Process 

Step 1: fork openclip repository https://github.com/mlfoundations/open_clip and navigate to open_clip/src/open_clip and move model_train.py file inside 

Step 2: to train model enter the command below and fill in with files specified with <> obtained from Step 3 of data generation process as well as path for saving the checkpoint of the model   
python <creative train captions dist 1> <creative train captions dist 2> <creative train captions dist 3> <creative train captions dist 4> <creative train captions dist 5> <creative val captions dist 1> <creative val captions dist 2> <creative val captions dist 3> <creative val captions dist 4> <creative val captions dist 5> <creative all train captions file> <creative all val captions file> <checkpoint saving path>

# Model Evaluation

Task 1: 

For task 1: download corpora from here: https://github.com/researchmm/img2poem/blob/master/data/multim_poem.json 

python eval_poem.py <baseline (true or false)> <checkpt path (None or path of checkpoint model from model training> <path to json file of corpora obtained above> <path to images directory> <task specified by distance (int 0-4) or None if baseline> 

Task 2: 

For task 2: download corpora here: 
https://github.com/tuhinjubcse/VisualMetaphors, there should be 6 different visual metaphor corpora downloaded each

python eval_met_task2.py <true or false for baseline clip or our clip model> <checkpt path of model or None if baseline> <metaphor directory 1> <metaphor directory 2> <metaphor directory 3> <metaphor directory 4> <metaphor directory 5><<metaphor directory 6>

Task 3: 
python eval_met_task3.py <checkpt path of model or None if baseline> <true or false for baseline clip or our clip model> <path to metaphors file> 


Note: We will release our generated dataset (raw associations and caption files) and checkpointed models in the near future.
