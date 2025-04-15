Algae Density Prediction using Environmental Factors 🌿📈
This project focuses on building a deep learning model with PyTorch to predict algae density based on various environmental factors. The goal is to analyze and visualize how different variables (e.g., temperature, pH, nutrient levels) influence the growth and concentration of algae, with the broader aim of supporting ecological monitoring and water quality management.

📁 Project Structure
bash
复制
编辑
algae_prediction/
├── data/                   # Datasets used for training and testing
├── model/                  # PyTorch model definitions and training scripts
├── utils/                  # Utility functions (data loading, preprocessing, etc.)
├── results/                # Visualization outputs and saved model results
├── figures/                # Plot images for environmental factor interactions
├── main.py                 # Main script to train and evaluate the model
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation
🔍 Features
Multivariate Time Series Prediction: Uses environmental variables (e.g., water temperature, DO, TN, TP) to predict algae density.

PyTorch Deep Learning Models: Includes customized architectures for regression tasks.

Data Visualization: Uses matplotlib for detailed plots suitable for reports or academic papers.

Reproducibility: All key steps from preprocessing to evaluation are documented and scriptable.

🚀 Getting Started
1. Clone the repository
bash
复制
编辑
git clone https://github.com/525514353/algae_prediction.git
cd algae_prediction
2. Install dependencies
We recommend using a virtual environment:

bash
复制
编辑
pip install -r requirements.txt
3. Prepare your data
Place your dataset (e.g., CSV format) into the data/ directory. Ensure the columns match the format expected by the scripts in utils/data_loader.py.

4. Train the model
bash
复制
编辑
python main.py
The script will handle training, validation, and save the resulting model and plots in the results/ folder.

📊 Visualization Examples
Environmental factor vs. algae density plots

3D surface plots showing interaction effects

Loss curves during training


Effect of Temperature and Nutrients on Algae Density

📚 Requirements
Python 3.8+

PyTorch

pandas

matplotlib

seaborn

scikit-learn

Install them via:

bash
复制
编辑
pip install -r requirements.txt
📌 Future Work
Add support for LSTM/GRU time series forecasting.

Integration with real-time sensor data.

Model optimization and hyperparameter tuning.

✍️ Author
Developed by 525514353
Feel free to open an issue or contact for collaboration.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
