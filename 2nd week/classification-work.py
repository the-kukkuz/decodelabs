import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DataClassificationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("DecodeLabs - Data Classification AI | Project 2")
        self.root.geometry("900x700")
        self.root.config(bg="#f0f0f0")
        
        # Variables
        self.df = None
        self.model = None
        self.label_encoders = {}
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the main UI"""
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=80)
        header.pack(fill=tk.X)
        
        title = tk.Label(header, text="DecodeLabs - Data Classification AI", 
                        font=("Arial", 16, "bold"), fg="white", bg="#2c3e50")
        title.pack(pady=10)
        
        subtitle = tk.Label(header, text="Project 2: Supervised Learning Classification", 
                           font=("Arial", 10), fg="#ecf0f1", bg="#2c3e50")
        subtitle.pack()
        
        # Main Container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Upload Button
        btn_frame = tk.Frame(main_container, bg="#f0f0f0")
        btn_frame.pack(pady=20)
        
        btn_upload = tk.Button(btn_frame, text="📂 Upload Dataset", 
                              command=self.upload_and_train,
                              bg="#3498db", fg="white", font=("Arial", 12, "bold"),
                              width=30, padx=20, pady=15, cursor="hand2")
        btn_upload.pack()
        
        # Report Area
        report_label = tk.Label(main_container, text="📊 Training Report & Results", 
                               font=("Arial", 12, "bold"), bg="#f0f0f0")
        report_label.pack(anchor=tk.W, pady=(20, 5))
        
        # Text area for report
        text_frame = tk.Frame(main_container, bg="white", relief=tk.SUNKEN, bd=2)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.report_text = tk.Text(text_frame, font=("Courier", 10), 
                                   yscrollcommand=scrollbar.set, bg="white")
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.config(command=self.report_text.yview)
        
        # Initial message
        initial_msg = """
╔════════════════════════════════════════════════════════════════╗
║        Welcome to DecodeLabs - Data Classification Tool       ║
║                      Project 2: AI Classification              ║
╚════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
1. Click "Upload Dataset" button above
2. Select your CSV or Excel file
3. Model training will start automatically
4. Results and accuracy report will appear here

SUPPORTED FORMATS: CSV (.csv), Excel (.xlsx)

EXPECTED DATA FORMAT:
- Rows: Data samples
- Columns: Features and Target (last column)
- Target column: Will be automatically detected as last column

EXAMPLE DATASET STRUCTURE:
┌─────────────────────────────────────┐
│ Age │ Income │ Score │ Class       │
├─────────────────────────────────────┤
│ 25  │ 30000  │ 600   │ Approved    │
│ 45  │ 80000  │ 750   │ Rejected    │
│ 35  │ 50000  │ 650   │ Approved    │
└─────────────────────────────────────┘

Waiting for dataset upload...
        """
        self.report_text.insert(tk.END, initial_msg)
        self.report_text.config(state=tk.DISABLED)
    
    def upload_and_train(self):
        """Upload dataset and train model automatically"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load dataset
            if file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            else:
                self.df = pd.read_csv(file_path)
            
            # Clear previous text
            self.report_text.config(state=tk.NORMAL)
            self.report_text.delete(1.0, tk.END)
            
            # Display initial info
            report = f"""
╔════════════════════════════════════════════════════════════════╗
║                    TRAINING IN PROGRESS                        ║
╚════════════════════════════════════════════════════════════════╝

📂 DATASET LOADED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
File: {file_path.split('/')[-1]}
Total Samples: {len(self.df)}
Total Features: {len(self.df.columns)}
Column Names: {', '.join(self.df.columns.tolist())}

📋 DATA PREVIEW (First 5 rows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{self.df.head().to_string()}

⚙️ PREPROCESSING & TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Data loaded successfully
✓ Detecting target column (last column)
✓ Encoding categorical features
✓ Splitting data: 80% Training | 20% Testing
✓ Training Gaussian Naive Bayes model...
            """
            
            self.report_text.insert(tk.END, report)
            self.report_text.config(state=tk.DISABLED)
            self.root.update()
            
            # Train model
            self.train_model()
            
        except Exception as e:
            self.report_text.config(state=tk.NORMAL)
            self.report_text.insert(tk.END, f"\n❌ ERROR: {str(e)}")
            self.report_text.config(state=tk.DISABLED)
            messagebox.showerror("Error", f"Failed to process dataset:\n{str(e)}")
    
    def train_model(self):
        """Train the model and generate report"""
        try:
            # Identify target column (last column)
            target_column = self.df.columns[-1]
            
            # Separate features and target
            X = self.df.iloc[:, :-1].copy()
            y = self.df[target_column].copy()
            
            # Encode categorical features
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            
            # Encode target if categorical
            le_target = None
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                target_classes = le_target.classes_
            else:
                target_classes = np.unique(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = GaussianNB()
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Generate final report
            self.report_text.config(state=tk.NORMAL)
            self.report_text.insert(tk.END, "\n✓ Model trained successfully\n")
            
            report = f"""

╔════════════════════════════════════════════════════════════════╗
║                    TRAINING COMPLETED ✓                        ║
╚════════════════════════════════════════════════════════════════╝

📊 MODEL PERFORMANCE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCURACY: {accuracy*100:.2f}%
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-Score: {f1:.4f}

📈 DETAILED METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Test Samples: {len(y_test)}
Correct Predictions: {(y_pred == y_test).sum()}
Incorrect Predictions: {(y_pred != y_test).sum()}
Error Rate: {((y_pred != y_test).sum() / len(y_test) * 100):.2f}%

🎯 CONFUSION MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{self.format_confusion_matrix(cm, target_classes, le_target)}

📋 TRAINING DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm: Gaussian Naive Bayes
Training Samples: {len(X_train)} (80%)
Testing Samples: {len(X_test)} (20%)
Number of Features: {X_train.shape[1]}
Target Classes: {len(target_classes)}
Classes: {', '.join(map(str, target_classes))}

📝 INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• ACCURACY: {self.interpret_metric(accuracy)}
• PRECISION: {self.interpret_precision(precision)}
• RECALL: {self.interpret_recall(recall)}
• F1-SCORE: {self.interpret_f1(f1)}

╔════════════════════════════════════════════════════════════════╗
║                  Model Ready for Deployment                    ║
╚════════════════════════════════════════════════════════════════╝
            """
            
            self.report_text.insert(tk.END, report)
            self.report_text.config(state=tk.DISABLED)
            messagebox.showinfo("Success", f"✓ Training Complete!\n\nAccuracy: {accuracy*100:.2f}%")
            
        except Exception as e:
            self.report_text.config(state=tk.NORMAL)
            self.report_text.insert(tk.END, f"\n\n❌ ERROR DURING TRAINING: {str(e)}")
            self.report_text.config(state=tk.DISABLED)
            messagebox.showerror("Training Error", f"Error during model training:\n{str(e)}")
    
    def format_confusion_matrix(self, cm, classes, le_target):
        """Format confusion matrix for display"""
        if le_target:
            classes = le_target.classes_
        
        header = "        Predicted\n"
        header += "        " + "  ".join([f"{str(c)[:8]:>8}" for c in classes]) + "\n"
        
        lines = [header]
        lines.append("        " + "-" * (len(classes) * 10) + "\n")
        
        for i, actual_class in enumerate(classes):
            line = f"{str(actual_class)[:8]:>8}|"
            for j in range(len(classes)):
                line += f"{cm[i, j]:>9}"
            lines.append(line)
        
        return "Actual\n" + "".join(lines)
    
    def interpret_metric(self, accuracy):
        """Interpret accuracy score"""
        if accuracy >= 0.9:
            return f"Excellent ({accuracy*100:.2f}%) - Model performs very well"
        elif accuracy >= 0.8:
            return f"Good ({accuracy*100:.2f}%) - Model performs well"
        elif accuracy >= 0.7:
            return f"Fair ({accuracy*100:.2f}%) - Model performs acceptably"
        else:
            return f"Poor ({accuracy*100:.2f}%) - Consider improving features or data"
    
    def interpret_precision(self, precision):
        """Interpret precision score"""
        if precision >= 0.9:
            return f"Excellent ({precision:.4f}) - Few false positives"
        elif precision >= 0.8:
            return f"Good ({precision:.4f}) - Reliable positive predictions"
        else:
            return f"Fair ({precision:.4f}) - Some false positives present"
    
    def interpret_recall(self, recall):
        """Interpret recall score"""
        if recall >= 0.9:
            return f"Excellent ({recall:.4f}) - Catches most true positives"
        elif recall >= 0.8:
            return f"Good ({recall:.4f}) - Good positive detection rate"
        else:
            return f"Fair ({recall:.4f}) - Misses some true positives"
    
    def interpret_f1(self, f1):
        """Interpret F1 score"""
        if f1 >= 0.9:
            return f"Excellent ({f1:.4f}) - Well-balanced model"
        elif f1 >= 0.8:
            return f"Good ({f1:.4f}) - Good balance between precision and recall"
        else:
            return f"Fair ({f1:.4f}) - Consider optimization"

def main():
    root = tk.Tk()
    app = DataClassificationTool(root)
    root.mainloop()

if __name__ == "__main__":
    main()