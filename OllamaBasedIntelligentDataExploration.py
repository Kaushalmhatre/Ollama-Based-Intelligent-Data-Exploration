import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import os

# Function to Perform EDA and Generate Visualizations
def eda_analysis(file_path):
    if file_path is None:
        return "Please upload a CSV file.", []
        
    df = pd.read_csv(file_path)
    
    # Fill missing values with median for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing values with mode for categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Data Summary
    summary = df.describe(include='all').to_string()
    
    # Missing Values
    missing_values = df.isnull().sum().to_string()

    # Generate AI Insights
    insights = generate_ai_insights(summary)
    
    # Generate Data Visualizations
    plot_paths = generate_visualizations(df)
    
    return f"Data Loaded Successfully!\n\nSummary:\n{summary}\n\nMissing Values:\n{missing_values}\n\nAI Insights:\n{insights}", plot_paths

# AI-Powered Insights using Gemma 2
def generate_ai_insights(df_summary):
    prompt = f"Analyze the dataset summary and provide 3-5 key business insights in bullet points:\n\n{df_summary}"
    try:
        # Ensure you have run 'ollama pull gemma2:2b' in your terminal first
        response = ollama.chat(model="gemma2:2b", messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"AI Insight Error: {str(e)}. Make sure Ollama is running!"

# Function to Generate Data Visualizations 
def generate_visualizations(df):
    plot_paths = []
    
    # Histograms for Numeric Columns
    numeric_cols = df.select_dtypes(include=['number']).columns[:5]
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=30, kde=True, color="blue")
        plt.title(f"Distribution of {col}")
        path = f"{col}_distribution.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()
    
    # Correlation Heatmap
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(8,5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        path = "correlation_heatmap.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()

    return plot_paths

# Gradio Interface
demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type="filepath"),
    outputs=[gr.Textbox(label="EDA Report"), gr.Gallery(label="Data Visualizations")],
    title="📊 Local AI Data Analysis",
    description="Upload a CSV to analyze it using your local Ollama model."
)

if __name__ == "__main__":
    demo.launch(share=True)




