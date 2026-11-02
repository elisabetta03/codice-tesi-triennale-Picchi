"""Annotate perceived gender (M/F) and perceived racial group (W/NW) at image level."""
from pathlib import Path
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / 'image_annotations.csv'
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
IMAGE_MARGIN = 20

def find_images(root: Path):
    return sorted([p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS], key=lambda p: str(p).lower())

def load_existing_annotations():
    annotations = {}
    if not CSV_PATH.exists():
        return annotations
    with CSV_PATH.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get('relative_path', '').strip()
            if key:
                annotations[key] = {'filename': row.get('filename', '').strip(), 'relative_path': key, 'perceived_gender': row.get('perceived_gender', '').strip(), 'perceived_racial_group': row.get('perceived_racial_group', '').strip()}
    return annotations

def save_annotations(annotations):
    rows = sorted(annotations.values(), key=lambda r: r['relative_path'].lower())
    with CSV_PATH.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'relative_path', 'perceived_gender', 'perceived_racial_group'])
        writer.writeheader()
        writer.writerows(rows)

class AnnotationApp:

    def __init__(self, root, image_root, images):
        self.root = root
        self.image_root = image_root
        self.images = images
        self.annotations = load_existing_annotations()
        self.index = self.find_first_unfinished()
        self.current_gender = ''
        self.current_race = ''
        self.original_image = None
        self.photo = None
        self.resize_job = None
        self.root.title('Annotazione immagini - M/F e W/NW')
        self.root.geometry('1200x850')
        self.root.minsize(800, 600)
        self.build_ui()
        self.bind_keys()
        self.root.bind('<Configure>', self.on_window_resize)
        self.show_current_image()

    def build_ui(self):
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        info = tk.Frame(self.root)
        info.grid(row=0, column=0, sticky='ew', padx=12, pady=(10, 4))
        info.grid_columnconfigure(1, weight=1)
        self.progress_label = tk.Label(info, text='', font=('Segoe UI', 11, 'bold'))
        self.progress_label.grid(row=0, column=0, sticky='w')
        self.filename_label = tk.Label(info, text='', font=('Segoe UI', 9), anchor='e')
        self.filename_label.grid(row=0, column=1, sticky='e', padx=(20, 0))
        self.current_annotation_label = tk.Label(self.root, text='', font=('Segoe UI', 14, 'bold'), pady=4)
        self.current_annotation_label.grid(row=1, column=0, sticky='ew', padx=12)
        controls = tk.Frame(self.root)
        controls.grid(row=2, column=0, sticky='ew', padx=12, pady=(4, 6))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)
        gender_frame = tk.LabelFrame(controls, text='Genere percepito', padx=8, pady=6)
        gender_frame.grid(row=0, column=0, sticky='ew', padx=(0, 6))
        self.gender_var = tk.StringVar(value='')
        tk.Radiobutton(gender_frame, text='M  [M]', variable=self.gender_var, value='M', command=lambda: self.set_gender('M'), font=('Segoe UI', 11)).pack(side='left', expand=True, padx=12)
        tk.Radiobutton(gender_frame, text='F  [F]', variable=self.gender_var, value='F', command=lambda: self.set_gender('F'), font=('Segoe UI', 11)).pack(side='left', expand=True, padx=12)
        race_frame = tk.LabelFrame(controls, text='Appartenenza razziale percepita', padx=8, pady=6)
        race_frame.grid(row=0, column=1, sticky='ew', padx=(6, 0))
        self.race_var = tk.StringVar(value='')
        tk.Radiobutton(race_frame, text='W - White  [W]', variable=self.race_var, value='W', command=lambda: self.set_race('W'), font=('Segoe UI', 11)).pack(side='left', expand=True, padx=12)
        tk.Radiobutton(race_frame, text='NW - Non-White  [N]', variable=self.race_var, value='NW', command=lambda: self.set_race('NW'), font=('Segoe UI', 11)).pack(side='left', expand=True, padx=12)
        self.image_frame = tk.Frame(self.root, bg='#222222')
        self.image_frame.grid(row=3, column=0, sticky='nsew', padx=12, pady=4)
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_label = tk.Label(self.image_frame, bg='#222222')
        self.image_label.grid(row=0, column=0, sticky='nsew')
        nav = tk.Frame(self.root)
        nav.grid(row=4, column=0, sticky='ew', padx=12, pady=(6, 4))
        nav.grid_columnconfigure(1, weight=1)
        tk.Button(nav, text='←  Precedente', command=self.previous_image, width=18).grid(row=0, column=0, sticky='w')
        self.status_label = tk.Label(nav, text='', font=('Segoe UI', 10))
        self.status_label.grid(row=0, column=1)
        tk.Button(nav, text='Successiva  →', command=self.next_image, width=18).grid(row=0, column=2, sticky='e')
        tk.Label(self.root, text='M/F = genere · W/N = White/Non-White · ←/→ = precedente/successiva · salvataggio automatico', font=('Segoe UI', 9), fg='#555555').grid(row=5, column=0, pady=(0, 8))

    def bind_keys(self):
        self.root.bind('<KeyPress-m>', lambda e: self.set_gender('M'))
        self.root.bind('<KeyPress-M>', lambda e: self.set_gender('M'))
        self.root.bind('<KeyPress-f>', lambda e: self.set_gender('F'))
        self.root.bind('<KeyPress-F>', lambda e: self.set_gender('F'))
        self.root.bind('<KeyPress-w>', lambda e: self.set_race('W'))
        self.root.bind('<KeyPress-W>', lambda e: self.set_race('W'))
        self.root.bind('<KeyPress-n>', lambda e: self.set_race('NW'))
        self.root.bind('<KeyPress-N>', lambda e: self.set_race('NW'))
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())

    def relative_key(self, image_path: Path):
        return str(image_path.relative_to(self.image_root)).replace('\\', '/')

    def annotation_is_complete(self, key):
        row = self.annotations.get(key)
        return bool(row and row.get('perceived_gender') in {'M', 'F'} and (row.get('perceived_racial_group') in {'W', 'NW'}))

    def find_first_unfinished(self):
        for i, image_path in enumerate(self.images):
            if not self.annotation_is_complete(self.relative_key(image_path)):
                return i
        return max(0, len(self.images) - 1)

    def completed_count(self):
        return sum((1 for image_path in self.images if self.annotation_is_complete(self.relative_key(image_path))))

    def save_current(self):
        image_path = self.images[self.index]
        key = self.relative_key(image_path)
        self.annotations[key] = {'filename': image_path.name, 'relative_path': key, 'perceived_gender': self.current_gender, 'perceived_racial_group': self.current_race}
        save_annotations(self.annotations)
        self.update_status()

    def set_gender(self, value):
        self.current_gender = value
        self.gender_var.set(value)
        self.save_current()
        self.auto_advance_if_complete()

    def set_race(self, value):
        self.current_race = value
        self.race_var.set(value)
        self.save_current()
        self.auto_advance_if_complete()

    def auto_advance_if_complete(self):
        if self.current_gender in {'M', 'F'} and self.current_race in {'W', 'NW'}:
            self.root.after(220, self.next_image)

    def previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_current_image()

    def next_image(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.show_current_image()
            return
        completed = self.completed_count()
        if completed == len(self.images):
            messagebox.showinfo('Annotazione completata', f'Tutte le {len(self.images)} immagini sono state annotate.\n\nCSV salvato in:\n{CSV_PATH}')
        else:
            messagebox.showinfo('Ultima immagine', "Sei arrivata all'ultima immagine, ma alcune annotazioni risultano incomplete.")

    def show_current_image(self):
        image_path = self.images[self.index]
        key = self.relative_key(image_path)
        row = self.annotations.get(key, {})
        self.current_gender = row.get('perceived_gender', '')
        self.current_race = row.get('perceived_racial_group', '')
        self.gender_var.set(self.current_gender)
        self.race_var.set(self.current_race)
        try:
            with Image.open(image_path) as img:
                self.original_image = img.convert('RGB').copy()
            self.root.after(20, self.render_current_image)
        except Exception as exc:
            self.original_image = None
            self.image_label.configure(image='', text=f"Errore durante l'apertura dell'immagine:\n{exc}", fg='white', font=('Segoe UI', 12))
        self.filename_label.configure(text=key)
        self.update_status()

    def render_current_image(self):
        if self.original_image is None:
            return
        self.image_frame.update_idletasks()
        available_width = max(100, self.image_frame.winfo_width() - IMAGE_MARGIN)
        available_height = max(100, self.image_frame.winfo_height() - IMAGE_MARGIN)
        img = self.original_image.copy()
        img.thumbnail((available_width, available_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo, text='')

    def on_window_resize(self, event):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(100, self.render_current_image)

    def update_status(self):
        completed = self.completed_count()
        self.progress_label.configure(text=f'Immagine {self.index + 1}/{len(self.images)} · Completate {completed}/{len(self.images)}')
        gender = self.current_gender or 'NON INSERITO'
        race = self.current_race or 'NON INSERITO'
        self.current_annotation_label.configure(text=f'Annotazione corrente:   {gender}   |   {race}')
        key = self.relative_key(self.images[self.index])
        if self.annotation_is_complete(key):
            self.status_label.configure(text='Salvata')
        elif self.current_gender or self.current_race:
            self.status_label.configure(text='Annotazione incompleta')
        else:
            self.status_label.configure(text='Da annotare')

def main():
    root = tk.Tk()
    root.withdraw()
    selected_folder = filedialog.askdirectory(title='Seleziona la cartella contenente le immagini')
    if not selected_folder:
        root.destroy()
        return
    image_root = Path(selected_folder)
    images = find_images(image_root)
    if not images:
        messagebox.showerror('Nessuna immagine trovata', 'La cartella selezionata non contiene immagini supportate.')
        root.destroy()
        return
    root.deiconify()
    AnnotationApp(root=root, image_root=image_root, images=images)
    root.mainloop()
if __name__ == '__main__':
    main()
