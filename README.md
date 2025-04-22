# Automated Generation and Validation of Financial Descriptions for Lithuanian Companies

This repository provides a complete solution for generating, validating, and translating financial descriptions of Lithuanian companies using local Large Language Models (LLMs). This project was created by Antanas Baltrušaitis and presented at PyCon 2025 in a talk titled "[Leveraging Large Language Models for Automated Generation and Validation of Financial Descriptions for Lithuanian Companies](slides/scoris_pycon2025_slides.pdf)".

## 🌟 Solution Overview
This project enables automated generation, validation, and multilingual translation (English ↔️ Lithuanian) of financial descriptions efficiently and accurately using open source and fine-tuned Large Language Models:

- **Generation**: Produce accurate and SEO-friendly financial descriptions based on raw financial data.
- **Validation**: Automatically verify the correctness of generated descriptions against original financial datasets.
- **Translation**: Translate descriptions seamlessly between English and Lithuanian leveraging powerful fine-tuned translation models.
- **Model Optimization**: Convert pre-trained translation models to optimized ONNX format for enhanced performance.

## 🚀 Solution Components

- **Generation (`generate.py`)**: Asynchronously fetch financial data, generate descriptions using an LLM via an API, and store results in a database.
- **Validation (`validate.py`)**: Asynchronously validate generated descriptions for accuracy against original data using an LLM API.
- **Translation (`translate.py`)**: Translate financial descriptions from English to Lithuanian using an efficient ONNX-optimized Seq2Seq transformer model.
- **Model Conversion (`convert_to_onnx.py`)**: Convert Hugging Face translation models into optimized ONNX format for improved inference speed.

## 📌 Models and Resources
Leveraging custom fine-tuned Hugging Face translation models for specific English-Lithuanian translation needs:
- [**EN → LT**: scoris-mt-en-lt](https://huggingface.co/scoris/scoris-mt-en-lt)
- [**LT → EN**: scoris-mt-lt-en](https://huggingface.co/scoris/scoris-mt-lt-en)

## 🙋 About Me
![Antanas Baltrušaitis](https://scoris.lt/images/apie-mus/antanas.webp)

**Antanas Baltrušaitis**  
- Founder @ [Scoris](https://scoris.lt) - Open Lithuanian Business Data Portal  
- Creator @ [Oriux](https://oriux.lt/programele/) – Advanced Lithuanian Weather App  
- Senior Analytics Engineer @ Beyond Analysis  
- Enthusiast of Open Data and AI  
📫 Connect on [LinkedIn](https://www.linkedin.com/in/abaltrusaitis/) | Personal website [baltrusaitis.lt](https://baltrusaitis.lt/)

## 🎙️ PyCon 2025 Conference Talk
**Topic**: [Leveraging Large Language Models for Automated Generation and Validation of Financial Descriptions for Lithuanian Companies](slides/scoris_pycon2025_slides.pdf)

## 📊 Use Cases & Achievements
- Generated and translated financial descriptions for over 100,000 Lithuanian companies using local LLMs.
- Enhanced accuracy, scalability, and SEO quality of financial content at [Scoris](https://scoris.lt).

## 🛠️ Technologies Used
- Python, asyncio, aiohttp, pandas, pyodbc
- Hugging Face, Transformers, and Optimum ONNX
- Aphrodite Engine and local GPTQ quantized LLM models

## 🔗 Related Services
- 🔎 [Scoris](https://scoris.lt): Open Lithuanian Business Data Portal providing free business information and reports.
- 🌤️ [Oriux](https://oriux.lt/programele/): Your best Lithuanian Weather Companion App.

---

©️ 2025 Antanas Baltrušaitis • [Scoris.lt](https://scoris.lt)