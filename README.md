Copy🪖 M.E.A.A — Military Environment Analysis Algorithm
G009 | DSTA Competition Project

Analyse any battlefield terrain in seconds — powered by computer vision and generative AI.


Overview
M.E.A.A (Military Environment Analysis Algorithm) is an AI-powered Streamlit application that assists military operational planners by analysing terrain images and generating environment-specific battle strategies, equipment recommendations, and optimal weaponry suggestions.
Developed for the DSTA competition, M.E.A.A combines custom-trained image classification models (via Google Teachable Machine) with OpenAI's generative AI to deliver a fast, holistic, and actionable analysis of any battlefield environment — from a single uploaded photo.

Features

🗺️ Terrain Classification — Identifies terrain type, landscape, and snow presence using three independent ML models
🤖 AI-Generated Strategies — Produces tailored battlefield strategies, equipment lists, and weaponry recommendations via GPT
📷 Live Image Capture — Supports real-time camera input for on-the-ground analysis
🏥 Health Scheduling — Provides optimal health and sleep schedules for soldiers based on operational context
🖥️ Clean Streamlit UI — Intuitive tabbed interface with sidebar navigation


How It Works
1. Terrain Classification
Three separate Teachable Machine models classify an uploaded image across three parameters:

Terrain (e.g. urban, forest, desert)
Landscape (e.g. mountainous, flat, coastal)
Snow Presence (snow / no snow)

Each model was trained on 20+ images per class sourced from open platforms like Pexels to ensure classification reliability.
2. AI Strategy Generation
The classified terrain description is passed into three separate GPT prompts to generate:

Battlefield strategies suited to the environment
Equipment required for soldiers
Optimal weaponry for the terrain

3. App Interface
Built with Streamlit, the app offers four navigation modes:
ModeDescriptionHomeIntroduction, how it works, and future expansionsUploadUpload a terrain image for full analysisLive-Image CaptureUse device camera for real-time terrain inputHealth SchedulingSoldier health and sleep recommendations

Tech Stack
ComponentTechnologyApp FrameworkStreamlitImage ClassificationTensorFlow / Keras (Teachable Machine)Generative AIOpenAI API (text-davinci-003)Image ProcessingPillow (PIL)Numerical ProcessingNumPy

Setup & Installation
bash# Clone the repository
git clone https://github.com/your-username/meaa.git
cd meaa

# Install dependencies
pip install streamlit tensorflow pillow numpy openai streamlit-option-menu

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the app
streamlit run app.py
Required Files
Ensure the following model and asset files are in the root directory:
terrain_model.h5
terrain_labels.txt
landscape_model.h5
landscape_labels.txt
snownosnow_model.h5
snownosnow_labels.txt
MEAA.png

Future Improvements

More Environmental Factors — Incorporate temperature, weather, and wind conditions
Real-Time Data Integration — Pull live weather data from external APIs for up-to-date analysis
Expanded Functionality — Add training programme generation based on recommended combat strategy
Enhanced ML Models — Train on larger, more diverse datasets for higher classification accuracy
Classified Data Training — In a real deployment, models would be trained on verified historical military operations


Team
Developed by Group 9 for the DSTA Competition.

All rights reserved: Group 9
