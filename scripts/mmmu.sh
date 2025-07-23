python main.py translate --hf-dataset MMMU/MMMU --hf-config Psychology --hf-split dev --target-lang mk --columns 'question','options' --source-lang en --backend nllb --streaming-mode --batch-size 100

# be careful with batch processing -- it will take a lot of time and might cost you a lot!
#python main.py translate --hf-dataset  OpenGVLab/ShareGPT-4o --hf-config image_caption --hf-split images --columns 'conversations.value' --target-lang mk --source-lang en --backend openai --use-batch
