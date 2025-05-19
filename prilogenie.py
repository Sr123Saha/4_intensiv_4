import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

class CommentClassifierApp:
    def __init__(self, root, model_path):
        self.root = root
        self.model_path = model_path
        
        self.class_names = [
           'Вопрос решен',
           'Нравится качество выполнения заявки',
           'Нравится качество работы сотрудников',
           'Нравится скорость отработки заявок',
           'Понравилось выполнение заявки',
           'Проблемы'
        ]
        
        self.threshold = 0.5
        self.default_font = ('Arial', 14)
        self.title_font = ('Arial', 16, 'bold')
        self.button_font = ('Arial', 14)
        self.tree_font = ('Arial', 13)
        
        self.load_model()
        self.setup_ui()
    
    def load_model(self):
        """Загружает только модель без label_map.json"""
        try:
            required_files = [
                "pytorch_model.bin",
                "config.json",
                "tokenizer_config.json"
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(self.model_path, file)):
                    raise FileNotFoundError(f"Не найден файл: {file}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            
            print("Модель успешно загружена!")
            print(f"Используемые классы: {self.class_names}")
            print(f"Порог классификации: {self.threshold}")
            
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить модель:\n{str(e)}")
            self.root.destroy()
    
    def setup_ui(self):
        """Настраивает интерфейс приложения"""
        self.root.title("Классификатор комментариев")
        self.root.geometry("900x700")
        
        style = ttk.Style()
        style.configure('.', font=self.default_font)
        style.configure('TLabel', font=self.default_font)
        style.configure('TButton', font=self.button_font)
        style.configure('Treeview', font=self.tree_font)
        style.configure('Treeview.Heading', font=self.tree_font)
        
        self.center_window()
        
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Введите ваш комментарий:", font=self.title_font).pack(pady=10)
        
        self.comment_input = scrolledtext.ScrolledText(
            main_frame, 
            wrap=tk.WORD,
            width=80,
            height=10,
            font=self.default_font
        )
        self.comment_input.pack(pady=15)
        
        classify_btn = ttk.Button(
            main_frame,
            text="Определить категории",
            command=self.classify_comment,
            style='TButton'
        )
        classify_btn.pack(pady=15)
        
        results_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="15", style='TLabel')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_tree = ttk.Treeview(
            results_frame,
            columns=('class', 'probability', 'is_selected'),
            show='headings',
            style='Treeview'
        )
        
        self.results_tree.heading('class', text='Категория')
        self.results_tree.heading('probability', text='Вероятность (%)')
        self.results_tree.heading('is_selected', text='Принадлежит')
        
        self.results_tree.column('class', width=350, anchor='w')
        self.results_tree.column('probability', width=250, anchor='center')
        self.results_tree.column('is_selected', width=200, anchor='center')
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_tree.tag_configure('selected', background='#e6f3e6')
        self.results_tree.tag_configure('not_selected', background='#f3e6e6')
    
    def center_window(self):
        """Центрирует окно на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def classify_comment(self):
        """Классифицирует введенный комментарий"""
        comment = self.comment_input.get("1.0", tk.END).strip()
        
        if not comment:
            messagebox.showwarning("Внимание", "Пожалуйста, введите комментарий")
            return
        
        try:
            inputs = self.tokenizer(
                comment,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits).numpy().flatten()
            
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            for class_name, prob in zip(self.class_names, probabilities):
                is_selected = "Да" if prob > self.threshold else "Нет"
                self.results_tree.insert(
                    '', 
                    'end', 
                    values=(
                        class_name,
                        f"{prob*100:.2f}%",
                        is_selected
                    ),
                    tags=('selected' if prob > self.threshold else 'not_selected')
                )
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при классификации:\n{str(e)}")

if __name__ == "__main__":
    MODEL_PATH = "./saved_comment_classifier"

    root = tk.Tk()
    app = CommentClassifierApp(root, MODEL_PATH)
    root.mainloop()