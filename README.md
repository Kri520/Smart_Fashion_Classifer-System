👗 Fashion AI Stylist — CNN-Powered Fashion Classification & Style Recommendation

From Garment Photo to Personal Stylist

Fashion AI Stylist is an AI-powered fashion application that goes beyond simple garment classification. A user uploads a garment image, and the system identifies the garment using a CNN, provides Top-3 predictions with confidence scores, analyzes visual properties such as dominant color, RGB value, brightness, and sharpness, and converts the result into outfit recommendations, occasion-based suggestions, fashion tips, and similar fashion looks.

The CNN model is trained and exported from a Jupyter Notebook, while the application uses the trained model for inference and combines it with a recommendation layer.

📌 Project Overview

The original classifier answers:

"What garment is this?"

The Fashion AI Stylist extends this into:

"What garment is this, what does it look like, how confident is the model, how can I style it, and where can I wear it?"

A single garment photo can trigger the complete pipeline:

Garment Image
     ↓
CNN Classification
     ↓
Top-3 Predictions + Confidence
     ↓
Visual Image Analysis
     ├── Dominant Color
     ├── RGB Value
     ├── Brightness
     └── Sharpness
     ↓
Style Recommendation Engine
     ├── Outfit Recommendation
     ├── Occasion Selection
     └── Fashion Tips
     ↓
Similar Fashion Images

🎯 Objectives

The main objectives of this project are:

Classify garment/fashion images using a CNN.

Provide more than one possible prediction using Top-3 classification.

Display confidence scores for predictions.

Analyze the visual properties of an uploaded image.

Extract the dominant garment color and RGB value.

Measure image brightness.

Measure image sharpness.

Recommend complementary outfit combinations.

Provide occasion-based fashion suggestions.

Generate short practical fashion tips.

Retrieve visually similar fashion images.

Provide all major results from a single image upload.

Keep model training and application inference modular and separate.

✨ Key Features

1. 🧠 CNN Fashion Classification

A Deep CNN trained on garment images predicts the fashion category.

The model uses four convolution + max-pooling blocks with increasing filter depth:

32 → 64 → 128 → 256 filters

2. 🏆 Top-3 Predictions

Instead of displaying only one prediction, the system returns the three most likely garment classes.

Example:

#1 Shirt       78%
#2 Coat        14%
#3 Pullover     6%

This provides additional information when visually similar garment classes are difficult to distinguish.

3. 📊 Confidence Score

The application displays the softmax probability associated with each Top-3 prediction.

Example:

Shirt      → 78%
Coat       → 14%
Pullover   →  6%

The highest-ranked prediction is used as the primary result for downstream styling recommendations.

4. 🎨 Dominant Color + RGB

The system analyzes image pixels to identify the dominant/representative color.

The result can be reported as an RGB triplet.

Example:

RGB(124, 74, 59)

This color information can be used as styling metadata and can support future color-matching functionality.

5. 💡 Brightness Analysis

The application analyzes image brightness to identify whether the uploaded image has suitable exposure.

Example:

Brightness: 72/100

Brightness analysis can help flag images that are too dark or too bright.

6. 🔍 Sharpness Analysis

The system also evaluates image sharpness.

Example:

Sharpness: 85/100

A low sharpness score can indicate a blurry image that may affect classification or visual analysis.

7. 👕 Outfit Recommendation

After identifying the garment, a rules-based recommendation layer suggests complementary pieces.

Possible recommendations can include:

Bottom wear

Layers

Accessories

Complementary clothing

Suitable footwear

The recommendation is based on the detected garment category.

8. 🎯 Occasion Selection

The system maps the detected garment to suitable occasions.

Supported recommendation contexts can include:

Casual

Formal

Festive

Work

The system can therefore answer not only:

What garment is this?

but also:

Where can I wear it?

9. 💬 Fashion Tips

The application provides short and practical styling guidance.

Examples of advice areas include:

Fit

Layering

Color pairing

Clothing combinations

Styling choices

10. 🖼️ Similar Fashion Images

The application can surface visually similar garments from the reference set.

This gives users additional fashion inspiration related to the uploaded image.

11. 📤 Simple Image Upload

The application provides a simple image upload interface.

Upload Garment Image
        ↓
Automatic Analysis
        ↓
Classification + Visual Analysis + Recommendations

No manual garment tagging is required.

12. 📓 Jupyter-Trained Model

The CNN is trained in a Jupyter Notebook and exported for application use.

This separates:

Model Training
      ↓
Model Export
      ↓
Application Inference

The application does not need to retrain the CNN for every prediction.

🔄 Complete Application Pipeline

┌─────────────────────────┐
│   Upload Garment Image  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    CNN Inference        │
│ Jupyter-Trained Model   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Top-3 Predictions       │
│ + Confidence Scores     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Visual Image Analysis   │
│                         │
│ • Dominant Color        │
│ • RGB                   │
│ • Brightness            │
│ • Sharpness             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Style Recommendation    │
│ Engine                  │
│                         │
│ • Outfit                │
│ • Occasion              │
│ • Fashion Tips          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Similar Fashion Images  │
└─────────────────────────┘

A single upload therefore triggers classification, image-quality analysis, and styling recommendations in one workflow.

🏗️ System Architecture

                       USER
                        │
                        ▼
              ┌──────────────────┐
              │  Image Upload    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Image Processing │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   CNN Model      │
              │  Classification  │
              └────────┬─────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Top-3 Predictions   │
             │ + Confidence       │
             └─────────┬──────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Color        Brightness    Sharpness
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Style Engine    │
              └────────┬─────────┘
                       │
             ┌─────────┼──────────┐
             │         │          │
             ▼         ▼          ▼
          Outfit    Occasion   Fashion Tips
       Recommendation Selection
             │         │          │
             └─────────┼──────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Similar Fashion  │
              │ Images / Looks   │
              └──────────────────┘

🧠 CNN Model Architecture

The CNN is defined and trained using the Keras Sequential API.

The image input is:

128 × 128 × 3 RGB

Images are rescaled using:

1 / 255

The network contains four convolution + max-pooling blocks.

Input
128 × 128 × 3
      ↓
Conv2D – 32 filters
      ↓
MaxPooling2D
      ↓
Conv2D – 64 filters
      ↓
MaxPooling2D
      ↓
Conv2D – 128 filters
      ↓
MaxPooling2D
      ↓
Conv2D – 256 filters
      ↓
MaxPooling2D
      ↓
Dense(128, ReLU)
      ↓
Dropout(0.4)
      ↓
Softmax Output

🧩 Model Architecture Details

Component

Configuration

Input

128 × 128 × 3 RGB

Input Scaling

1/255

Convolution Blocks

4

Conv Filters

32 → 64 → 128 → 256

Pooling

MaxPooling2D

Dense Layer

128 neurons

Activation

ReLU

Dropout

0.4

Output

Softmax

Total Parameters

2,486,986

Trainable Parameters

2,486,986

Non-Trainable Parameters

0

Model Size

9.49 MB

Number of Layers

13

🔬 Model Prediction Engine

The CNN's softmax layer produces a probability for each supported garment class.

Instead of returning only one class:

Prediction → Shirt

the application displays:

#1 Shirt       78%
#2 Coat        14%
#3 Pullover     6%

This Top-3 approach is useful for visually similar categories.

The highest-ranked prediction is passed to the recommendation layer for downstream styling guidance.

🎨 Image Intelligence

The system performs image-level analysis in addition to CNN classification.

Dominant Color

The image is analyzed to determine a representative color.

Example RGB values:

RGB(124, 74, 59)
RGB(47, 60, 85)
RGB(184, 137, 43)

Brightness

Example:

Brightness: 72/100

Sharpness

Example:

Sharpness: 85/100

These values provide additional information about the uploaded image and can also be used for future recommendation improvements.

👔 Style Recommendation Engine

Once the garment is identified, the system uses a rules-based recommendation layer to convert the prediction into practical fashion guidance.

The recommendation engine can provide:

Detected Garment
      ↓
Complementary Clothing
      ↓
Suitable Occasion
      ↓
Fashion Tips

For example:

Detected Garment:
Shirt

        ↓

Outfit:
Shirt + Suitable Bottom + Footwear

        ↓

Occasion:
Casual / Formal / Work / Festive

        ↓

Fashion Tip:
Fit, layering or color-pairing advice

🎯 Occasion-Based Recommendations

The style assistant can map garments to different contexts.

Casual

Suggestions can focus on comfortable everyday combinations.

Formal

Suggestions can focus on formal clothing, structured combinations, and suitable footwear.

Festive

Suggestions can focus on festive or occasion-appropriate styling.

Work

Suggestions can focus on professional and work-appropriate combinations.

The exact recommendation depends on the detected garment and the rules implemented in the application.

🖼️ Similar Fashion Discovery

The system can retrieve visually similar garments from the available reference set.

Workflow:

Uploaded Image
      ↓
Visual Representation / Reference Comparison
      ↓
Similar Fashion Images
      ↓
Related Looks / Inspiration

This feature allows users to discover similar fashion styles rather than receiving only a classification label.

📁 Project Structure

The project repository is organized as follows:

Smart_Fashion_Classifier-System/
│
├── 📂 dataset/
│   └── Fashion image dataset
│
├── 📂 models/
│   └── Trained CNN model
│
├── 📂 outputs/
│   └── Generated prediction / application outputs
│
├── 📂 uploads/
│   └── Uploaded garment images
│
├── 📓 SmartFashion_AI_Project_Corrected.ipynb
│   └── CNN training, evaluation and experimentation
│
├── 🐍 app.py
│   └── Main application and styling workflow
│
├── 📄 requirements.txt
│   └── Required Python packages
│
└── 📖 README.md
    └── Project documentation

📂 Folder Explanation

dataset/

Contains the fashion/garment image data used to train and evaluate the CNN.

models/

Contains the trained model exported from the Jupyter Notebook.

The model is used by the application for inference.

uploads/

Contains garment images uploaded through the application when the application is configured to store them.

outputs/

Contains generated outputs or prediction-related files when applicable.

SmartFashion_AI_Project_Corrected.ipynb

The notebook contains the model development workflow:

Import Libraries
      ↓
Load Dataset
      ↓
Preprocess Images
      ↓
Build CNN
      ↓
Compile Model
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Visualize Results
      ↓
Export Model

app.py

The application layer responsible for:

Image upload

CNN inference

Top-3 predictions

Confidence display

Color analysis

Brightness analysis

Sharpness analysis

Outfit recommendations

Occasion selection

Fashion tips

Similar fashion image discovery

requirements.txt

Contains the dependencies required to run the project.

🛠️ Technology Stack

Technology

Purpose

Python

Core programming language

TensorFlow / Keras

CNN definition, training and inference

NumPy

Array operations, image tensors and normalization

OpenCV / Pillow

Image processing, color, brightness and sharpness analysis

Matplotlib

Training curves, confusion matrix and sample predictions

Jupyter / Colab

Model training and experimentation

Streamlit

Interactive application interface

Git / GitHub

Version control and project hosting

🔄 Development Workflow

1. Problem Definition
        ↓
2. Fashion Dataset Preparation
        ↓
3. Image Preprocessing
        ↓
4. CNN Architecture Design
        ↓
5. Model Training in Jupyter
        ↓
6. Model Evaluation
        ↓
7. Model Export
        ↓
8. Streamlit Application
        ↓
9. Image Upload
        ↓
10. CNN Classification
        ↓
11. Visual Image Analysis
        ↓
12. Recommendation Engine
        ↓
13. Similar Fashion Discovery
        ↓
14. Testing
        ↓
15. GitHub Documentation

📦 Installation

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/Smart_Fashion_Classifier-System.git

Then:

cd Smart_Fashion_Classifier-System

Replace YOUR_USERNAME with your actual GitHub username.

2. Create a Virtual Environment

Windows

python -m venv venv

Activate:

venv\Scripts\activate

Linux / macOS

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

▶️ Train the Model

Open the Jupyter Notebook:

jupyter notebook

Then open:

SmartFashion_AI_Project_Corrected.ipynb

Run the training workflow and export/save the trained CNN model into the models/ directory.

🌐 Run the Application

After the trained model is available:

streamlit run app.py

If the streamlit command is not recognized:

python -m streamlit run app.py

Open the local Streamlit URL shown in the terminal, commonly:

http://localhost:8501

🖥️ How to Use

Step 1 — Upload

Upload a garment/fashion image.

Step 2 — Classification

The CNN predicts the most likely garment categories.

Step 3 — Review Top-3

View the three highest-ranked predictions and their confidence scores.

Step 4 — Analyze the Image

The system provides:

Dominant color

RGB value

Brightness

Sharpness

Step 5 — Get Styling Guidance

The recommendation engine provides:

Outfit recommendation

Occasion fit

Fashion tips

Step 6 — Discover Similar Looks

Browse visually similar fashion images from the reference set.

📊 Example Output

An example prediction shown by the presentation is:

Top-3 Predictions

#1  Shirt       78%
#2  Coat        14%
#3  Pullover     6%

Example image intelligence:

Dominant Color:
RGB(124, 74, 59)

Brightness:
72/100

Sharpness:
85/100

These are presentation examples demonstrating the application's output format.

📈 Model Performance

The project presentation documents the CNN architecture and model size, but it does not provide a final test/validation accuracy percentage.

Therefore, the final README should not invent an accuracy value.

Documented model information:

Total Parameters:       2,486,986
Trainable Parameters:  2,486,986
Non-Trainable Params:          0
Model Size:                  9.49 MB
Layers:                          13
Input:                  128×128×3 RGB

If a final training run provides accuracy, validation accuracy, loss, or a confusion matrix, those measured values can be added here.

📊 Recommended Evaluation Metrics

For future/final evaluation, the following metrics can be reported:

Accuracy

Validation Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Training Loss

Validation Loss

These should be filled with values obtained from the actual model evaluation rather than estimated values.

⚠️ Challenges & Limitations

1. Visually Similar Classes

Some garment classes have similar silhouettes.

Examples include:

Shirt
T-shirt / Top
Pullover
Coat

These similarities can lead to classification errors.

2. Lighting-Dependent Color

Dominant color and brightness measurements can change depending on the lighting conditions of the uploaded photograph.

Therefore, the detected RGB value may represent the image appearance rather than the exact physical garment color.

3. Recommendation Coverage

The recommendation layer covers common styling cases through rules.

Unusual or unsupported garments may receive a more generic recommendation.

4. Real-World Image Variation

Real user photographs may contain:

Different backgrounds

Different camera angles

Clutter

Different lighting

Different image quality

Different garment orientations

These conditions can differ from the training images and affect classification.

🚀 Future Scope

1. Data Augmentation

Introduce augmentation techniques such as:

Rotation

Horizontal flipping

Zoom

Additional transformations

This can help improve generalization to real-world photographs.

2. Transfer Learning

Experiment with pretrained models such as:

ResNet

MobileNet

Transfer learning can be explored to improve accuracy and inference efficiency.

3. Larger Fashion Dataset

Extend the dataset with:

More garment classes

Higher-resolution images

Real-world photographs

More diverse backgrounds

More lighting conditions

4. Advanced Personalization

Future versions can consider:

User style preference

Color preference

Season

Weather

Budget

Preferred clothing combinations

Accessories

Footwear

Recommendation history

5. Advanced Color Matching

The dominant RGB value can be combined with color compatibility rules to recommend matching clothing colors.

Possible workflow:

Detected Garment Color
        ↓
Color Compatibility Analysis
        ↓
Matching Colors
        ↓
Outfit Recommendation

6. Better Recommendation Engine

The current rules-based recommendation approach can be extended to a more advanced recommendation model using:

User preferences

Historical interactions

Outfit compatibility

Similarity

Context

Occasion

Color

Season

7. Deployment

The complete pipeline can be packaged behind:

A web application

A REST API

A cloud-hosted service

This would allow users to access the Fashion AI Stylist remotely.

🔐 Privacy Considerations

If the application is deployed publicly:

Avoid unnecessary permanent storage of uploaded images.

Protect uploaded user data.

Delete temporary files when they are no longer needed.

Avoid collecting personal information unless required.

Clearly communicate how images are processed.

🧪 Testing

The application should be tested using:

Image Tests

Clear garment photos

Blurry images

Dark images

Bright images

Different backgrounds

Different garment orientations

Different image sizes

Similar-looking garments

Recommendation Tests

Casual context

Formal context

Festive context

Work context

Different garment categories

The purpose is to evaluate both the classification pipeline and the recommendation behavior.

📷 Screenshots

Add screenshots of the actual application to the repository.

Recommended structure:

screenshots/
├── home.png
├── image-upload.png
├── top3-prediction.png
├── image-analysis.png
├── outfit-recommendation.png
├── occasion-recommendation.png
└── similar-fashion.png

Then include them in the README:

![Home](screenshots/home.png)

![Image Upload](screenshots/image-upload.png)

![Top-3 Prediction](screenshots/top3-prediction.png)

![Image Analysis](screenshots/image-analysis.png)

![Outfit Recommendation](screenshots/outfit-recommendation.png)

![Occasion Recommendation](screenshots/occasion-recommendation.png)

💡 Example End-to-End Scenario

Suppose the user uploads a garment image.

The application performs:

USER UPLOAD
     │
     ▼
Garment Image
     │
     ▼
CNN Classification
     │
     ├── Shirt     78%
     ├── Coat      14%
     └── Pullover   6%
     │
     ▼
Visual Analysis
     │
     ├── Dominant Color
     ├── RGB
     ├── Brightness
     └── Sharpness
     │
     ▼
Style Engine
     │
     ├── Outfit Recommendation
     ├── Occasion Selection
     └── Fashion Tips
     │
     ▼
Similar Fashion Images

The result is therefore much more than a classification label.

🌟 Why This Project Is Different

A conventional image classifier generally performs:

Image → Class

This project extends the workflow:

Image
  ↓
Class
  ↓
Confidence
  ↓
Visual Properties
  ↓
Style Guidance
  ↓
Occasion Guidance
  ↓
Fashion Tips
  ↓
Similar Looks

This makes the project a broader Fashion AI Stylist rather than only a garment classifier.

📚 Learning Outcomes

This project demonstrates practical experience with:

Python

Computer Vision

Image Classification

Deep Learning

CNN

TensorFlow

Keras

NumPy

Pillow / OpenCV

Matplotlib

Jupyter Notebook

Streamlit

Image preprocessing

Model training

Model evaluation

Model export

Model inference

Recommendation logic

Git

GitHub

Technical documentation

🧾 Project Components

Component

Purpose

Fashion Dataset

Training and evaluation images

Image Preprocessing

Converts images into model-ready input

CNN

Learns garment visual features

Top-3 Prediction

Provides multiple possible garment classes

Softmax

Produces confidence probabilities

Color Analysis

Extracts dominant color/RGB

Brightness Analysis

Measures image exposure

Sharpness Analysis

Measures image focus/quality

Style Engine

Generates fashion guidance

Outfit Recommendation

Suggests complementary clothing

Occasion Selection

Maps clothing to contexts

Fashion Tips

Provides practical styling advice

Similar Images

Provides related fashion looks

Jupyter Notebook

CNN training and experimentation

Streamlit

Interactive application

GitHub

Project version control and sharing

🏆 Project Highlights

✓ CNN-Based Fashion Classification
✓ Top-3 Ranked Predictions
✓ Softmax Confidence Scores
✓ Dominant Color Detection
✓ RGB Extraction
✓ Brightness Analysis
✓ Sharpness Analysis
✓ Outfit Recommendations
✓ Occasion-Based Suggestions
✓ Fashion Tips
✓ Similar Fashion Images
✓ Jupyter-Trained CNN
✓ Streamlit Application
✓ Modular and Extensible Architecture

🔮 Possible Extended Vision

The long-term vision of the project is:

             USER
              │
              ▼
        Upload Garment
              │
              ▼
       Fashion AI Stylist
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
   Identify  Analyze  Understand
   Garment   Image    Context
       │      │       │
       └──────┼───────┘
              │
              ▼
      Personalized Style
          Recommendation
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
     Outfit Occasion Tips
       │      │      │
       └──────┼──────┘
              ▼
        Similar Looks

👨‍💻 Author

Krishna Meena


⭐ Conclusion

Fashion AI Stylist extends a conventional garment classifier into a complete fashion assistance workflow.

A CNN trained in Jupyter Notebook provides Top-3 garment predictions with confidence scores. Additional image analysis extracts dominant color, RGB, brightness, and sharpness. A rules-based styling layer then converts the detected garment into outfit recommendations, occasion-based guidance, and practical fashion tips, while similar-image retrieval provides additional inspiration.

The complete system is designed around a single simple interaction:

              UPLOAD ONE IMAGE
                    ↓
        ┌─────────────────────────┐
        │    FASHION AI STYLIST   │
        └─────────────────────────┘
                    ↓
          Identify the Garment
                    ↓
           Show Top-3 Results
                    ↓
          Analyze Image Quality
                    ↓
         Understand Color / RGB
                    ↓
          Recommend an Outfit
                    ↓
          Suggest an Occasion
                    ↓
            Give Fashion Tips
                    ↓
           Show Similar Looks

Fashion AI Stylist — From Garment Photo to Personal Stylist.
