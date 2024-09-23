import os

# Укажите путь к вашей библиотеке
lib_path = project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ))  # Путь к вашей библиотеке
output_dir = lib_path + '/docs/source/modules'  # Директория для сохранения .rst файлов

# Получаем список всех файлов в директории библиотеки
modules = [f[:-3] for f in os.listdir(lib_path+'/ride') if f.endswith('.py') and f != '__init__.py']
print(os.listdir(lib_path))
# Генерируем .rst файлы для каждого модуля
for module in modules:
    rst_content = f"""\
{module.capitalize()} Module
==========================

.. automodule:: ride.{module}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    with open(output_dir+f"/{module}.rst", 'w', encoding='utf-8') as rst_file:
        rst_file.write(rst_content)

print("RST files generated successfully.")