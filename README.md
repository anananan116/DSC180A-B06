# DSC180A-B06 Quarter 1 Project

Reflection Based Reasoning Improvements to LLM Process Reward Modeling. Preceeding work to our Winter 2025 project, [Multi-Modal Reasoners](https://github.com/Lancelot39/mm-reasoners)

# Requirements
- Conda 
- Python >=3.10
- OpenAI account

# Setup (Local Development)

Navigate to the root directory, then setup the Conda Environment
```bash
$ conda env create -f environment.yml
```

Download the MATH dataset from
```url
https://github.com/hendrycks/math
```

then, put the unziped folder in ./data/

To generate math reasoning using small model, navigate to the root directory:
```bash
python main.py
```

- All datasets can be retrieved via HuggingFace.
- Make sure you create your own OpenAI API Key. Note: OpenAI's key only serves OpenAI models

## Structure
```
DSC180A-B06 (root)
├── data/                   # Contains datasets from HuggingFace/Github
├── config/                 # Config files
├── reports/                # Summaries on Model Performances by Round
│        
├── main.py                
├── inference.py              
├── eval.py
├── reflection.py
├── data_utils.py
├── environment.yml         # environment specification, dependencies
└── README.md               
```
