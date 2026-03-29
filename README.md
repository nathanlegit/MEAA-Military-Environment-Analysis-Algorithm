# M.E.A.A — Military Environment Analysis Algorithm
**G009 | DSTA Young Defence Scientist Programme 2023 · 🥈 First Runners Up**

> Analyse any battlefield terrain in seconds — powered by computer vision and generative AI.

---

## What is M.E.A.A?

M.E.A.A is an AI-powered web application that helps military operational planners make faster, smarter decisions about unfamiliar terrain. Upload a photo of any environment — desert, forest, snow, urban — and M.E.A.A will classify the terrain and instantly generate tailored battlefield strategies, equipment checklists, and weaponry recommendations.

Built for the DSTA Young Defence Scientist Programme, the system combines three custom-trained image classification models with OpenAI's generative AI to produce a complete operational analysis from a single image.

---

## Features

| Feature | Description |
|---|---|
| 🗺️ Terrain Classification | Identifies terrain type, landscape, and snow presence using three independent CNN models |
| 🤖 AI-Generated Strategies | Produces tailored battlefield strategies, equipment lists, and weaponry recommendations via GPT |
| 📷 Live Image Capture | Supports real-time camera input for on-the-ground analysis |
| 🏥 Health Scheduling | Provides health and sleep schedule recommendations for soldiers based on operational context |
| 🖥️ Streamlit UI | Intuitive tabbed interface with sidebar navigation |

---

## How It Works

### 1. Terrain Classification
Three separate Teachable Machine models classify an uploaded image across three parameters:
- **Terrain** — e.g. urban, forest, desert
- **Landscape** — e.g. mountainous, flat, coastal
- **Snow Presence** — snow or no snow

Each model was trained on 20+ images per class sourced from open platforms like Pexels. The three outputs are combined into a single terrain description string.

### 2. AI Strategy Generation
The terrain description is passed into three separate GPT prompts to generate:
- Battlefield strategies suited to the environment
- Equipment required for soldiers
- Optimal weaponry for the terrain

### 3. App Interface
The app has four navigation modes:

| Mode | Description |
|---|---|
| Home | Introduction, how it works, and future expansions |
| Upload | Upload a terrain image for full analysis |
| Live-Image Capture | Use your device camera for real-time terrain input |
| Health Scheduling | Soldier health and sleep recommendations |

---

## Tech Stack

| Component | Technology |
|---|---|
| App Framework | Streamlit |
| Image Classification | TensorFlow / Keras (Google Teachable Machine) |
| Generative AI | OpenAI API (text-davinci-003) |
| Image Processing | Pillow (PIL) |
| Numerical Processing | NumPy |

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- An OpenAI API key

### Steps


# Clone the repository
git clone https://github.com/nathanlegit/MEAA-Military-Environment-Analysis-Algorithm.git
cd MEAA-Military-Environment-Analysis-Algorithm

# Install dependencies
pip install streamlit tensorflow pillow numpy openai streamlit-option-menu

# Set your OpenAI API key (Mac/Linux)
export OPENAI_API_KEY="your-api-key-here"

# Set your OpenAI API key (Windows PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# Run the app
streamlit run app.py


### Repository Structure


MEAA-Military-Environment-Analysis-Algorithm/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md
│
├── models/                       # Trained CNN models and class labels
│   ├── terrain_model.h5
│   ├── terrain_labels.txt
│   ├── landscape_model.h5
│   ├── landscape_labels.txt
│   ├── snownosnow_model.h5
│   └── snownosnow_labels.txt
│
├── assets/                       # Images and UI assets
│   ├── MEAA.png
│   ├── sample.png
│   └── home_image1.jfif ... home_image7.png
│
└── tests/
    └── sample_images/            # Test images for validating model output
        ├── test_desert.jpg
        ├── test_forest.jpg
        ├── test_snow.jpg
        └── test_rural.jpg


---

## Future Improvements

- **More Environmental Factors** — Incorporate temperature, weather, and wind conditions into the analysis
- **Real-Time Data Integration** — Pull live weather data from external APIs for up-to-date recommendations
- **Expanded Functionality** — Add a training programme generator based on the recommended combat strategy
- **Enhanced ML Models** — Train on larger, more diverse datasets for higher classification accuracy
- **Classified Data Training** — In a real deployment, models would be trained on verified historical military operations
