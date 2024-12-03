# DSC180A-B06

# Requirements
- Conda 
- Python >=3.10
- OpenAI account 

# Setup

Setup the Conda Environment
```bash
conda env create reasoners
```

Download the MATH dataset from
```url
https://github.com/hendrycks/math
```

Put the unziped folder in ./data/

To generate math reasoning using small model:
```bash
python main.py
```

- All datasets can be retrieved via HuggingFace.
- Make sure you create your own OpenAI API Key. Note: OpenAI's key only serves OpenAI models
