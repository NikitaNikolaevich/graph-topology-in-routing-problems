import os

# Укажите путь к вашей библиотеке
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(lib_path, 'docs', 'source', 'modules')  # Директория для сохранения .rst файлов

# Создаем директорию для .rst файлов, если она не существует
os.makedirs(output_dir, exist_ok=True)

# Получаем список всех файлов в директории библиотеки
modules = [
    f[:-3] for f in os.listdir(os.path.join(lib_path, 'ride'))
    if f.endswith('.py') and f != '__init__.py'
]

# Генерируем .rst файлы для каждого модуля
for module in modules:
    rst_content = f"""{module.capitalize()} Module
==========================

.. automodule:: ride.{module}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    try:
        with open(os.path.join(output_dir, f"{module}.rst"), 'w', encoding='utf-8') as rst_file:
            rst_file.write(rst_content)
    except Exception as e:
        print(f"Error writing file {module}.rst: {e}")

print("RST files generated successfully.")
