# AI Research Agent System

A dual-model research automation system that combines web crawling with AI analysis to generate comprehensive research reports. Supports both cloud-based Hugging Face models and local LM Studio deployments.

## Key Features

- **Dual-Agent Architecture**
  - Research Agent: Performs web research using Tavily API
  - Drafting Agent: Structures findings into formal reports
- **Multi-Model Support**
  - Hugging Face Endpoint integration (cloud)
  - LM Studio local model deployment
- **Automated Workflow**
  - LangGraph-powered state management
  - PDF report generation
  - Error handling and output parsing

## System Components

1. `huggingFaceAPI_model.py` - Cloud-based version using Hugging Face models
2. `openAI_API_Model_LMstudio.py` - Local version using LM Studio
3. `lmstudio_api_test.py` - Validation script for local LM Studio setup
4. `.env` - API key configuration (not included in repo)
5. `requirements.txt` - Dependency list

## Workflow Process

1. **Input Handling**
   - Accepts natural language research queries
   - Initializes LangGraph state machine

2. **Research Phase**
   - Tavily API web crawling (10+ sources)
   - Source validation and credibility scoring
   - Data aggregation from multiple perspectives

3. **Analysis Phase**
   - FLAN-T5/Zephyr model processing
   - Fact extraction and correlation
   - Bias detection and mitigation

4. **Drafting Phase**
   - Structured report generation
   - Academic formatting enforcement
   - Source citation management

5. **Output Generation**
   - PDF export with proper formatting
   - Error-logged output validation
   - File system organization

## Installation

bash
# Clone repository
git clone [your-repo-url]
cd project-directory

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
Configuration
Obtain API keys:

Tavily: https://app.tavily.com/

Hugging Face: https://huggingface.co/settings/tokens

Edit .env:

ini
TAVILY_API_KEY=your_tavily_key
HUGGINGFACE_API_KEY=your_hf_token
For LM Studio:

Download model: https://lmstudio.ai/

Run local inference server on port 1234

Usage
Hugging Face Cloud Version
bash
python huggingFaceAPI_model.py
LM Studio Local Version
bash
python openAI_API_Model_LMstudio.py
LM Studio Validation
bash
python lmstudio_api_test.py
Requirements
Python 3.9+

Tavily API key (free tier available)

Hugging Face Hub access token

LM Studio (for local version)

4GB+ RAM for local model operation

Troubleshooting
Model Access Issues

Verify Hugging Face model permissions

Check LM Studio server status

API Errors

Validate .env file formatting

Confirm network connectivity

PDF Generation

Ensure write permissions in /output

Check for special characters in content

Security Notes
API keys never committed to version control

Local model operations stay on-device

Tavily API requests encrypted via HTTPS

Disclaimer
This system requires responsible use:

Verify generated facts with original sources

Adhere to target websites' robots.txt

Comply with Hugging Face's Model License

Monitor API usage quotas

For research purposes only. Not responsible for output accuracy.


This README provides:
1. Clear technical overview without jargon
2. Step-by-step setup instructions
3. Architecture documentation
4. Usage scenarios for both versions
5. Important compliance information

The structure balances conciseness with technical completeness while maintaining professional tone.