"""
General purpose utils files.

C: 2021.10.28 / U: 2021.11.02
"""

from pathlib import Path

def save_notebook_as_html(file_name, cleaning_delay=5):
    """
    Save the notebook as a html file in the `html_pdf` folder. 
    
    Parameters:
    ------------------------
    file_name - a notebook file_name with/without '.ipynb' extension.    
    C: 2021.10.28 / M: 2021.10.30
    """
    import os        
    import time
    from IPython.display import clear_output
    
    if file_name.endswith('.ipynb'):
        file_name = os.path.splitext(file_name)[0]
    
    os.system(f'jupyter nbconvert --to html {file_name}.ipynb')   
    print(f'*** Converting "{file_name}":\n\t ipynb   --->  .html')
    
    # move created html file to a html_pdf folder
    src = f'{file_name}.html'
    dst = f'html_pdf/{src}'
    dst_folder = Path('html_pdf')
    if not dst_folder.exists():
        dst_folder.mkdir(parents=True, exist_ok=True )
    os.rename(src, dst)
    print('Files saved in "html_pdf" folder')
    
    # clear cell output after some time
    time.sleep(cleaning_delay)
    clear_output(wait=False)
    time.sleep(1)
    
    
    

    
def save_notebook_as_pdf(file_name, cleaning_delay=5):
    """
    Convert the notebook to *.tex and pdf formats.
    
    Parameters:
    ------------------------
    file_name - a notebook file_name with/without '.ipynb' extension. 
    cleaning_delay - time to wait unitl a notebook cell output will be cleared.
    
    Files are saved in `html_pdf` folder.
    
    C: 2021.10.28 / M: 2021.10.30
    """
    import os 
    import shutil
    import time
    from IPython.display import clear_output

    # conversion to tex
    os.system(f'jupyter nbconvert --to latex {file_name}.ipynb')
    print(f'*** Converting "{file_name}" :\n\t ipynb   --->   tex')         
    
    # conversion to pdf
    os.system(f'pdflatex -interaction=nonstopmode {file_name}.tex')
    print(f'\t tex   --->  pdf')
              
    print('*** Cleaning (if neeaded):')
    
    # remove files
    ext = ['.tex', '.log', '.out', '.aux']
    for e in ext:
        file = file_name + e
        if os.path.exists(file):            
            os.remove(file)
            print(f'\t{file}')
    
    # remove automatically created folder 
    folder = file_name + '_files' 
    if os.path.exists(folder):        
        shutil.rmtree(folder, ignore_errors=True)
        print(f'\t{folder}')
        
    # move created pdf file to a html_pdf folder
    src = f'{file_name}.pdf'
    dst = f'html_pdf/{src}'
    os.rename(src, dst)
    print('Files saved in "html_pdf" folder')
    
    # clear cell output after some time
    time.sleep(cleaning_delay)
    clear_output(wait=False)
    
def save_notebook_as_pdf(file_name, cleaning_delay=5):
    """
    Convert the notebook to *.tex and pdf formats.
    
    Parameters:
    ------------------------
    file_name - a notebook file_name with/without '.ipynb' extension. 
    cleaning_delay - time to wait unitl a notebook cell output will be cleared.
    
    Files are saved in `html_pdf` folder.
    
    C: 2021.10.28 / M: 2021.10.30
    """
    import os 
    import shutil
    import time
    from IPython.display import clear_output

    # conversion to tex
    os.system(f'jupyter nbconvert --to latex {file_name}.ipynb')
    print(f'*** Converting "{file_name}" :\n\t ipynb   --->   tex')         
    
    # conversion to pdf
    os.system(f'pdflatex -interaction=nonstopmode {file_name}.tex')
    print(f'\t tex   --->  pdf')
              
    print('*** Cleaning (if neeaded):')
    
    # remove files
    ext = ['.tex', '.log', '.out', '.aux']
    for e in ext:
        file = file_name + e
        if os.path.exists(file):            
            os.remove(file)
            print(f'\t{file}')
    
    # remove automatically created folder 
    folder = file_name + '_files' 
    if os.path.exists(folder):        
        shutil.rmtree(folder, ignore_errors=True)
        print(f'\t{folder}')
        
    # move created pdf file to a html_pdf folder
    src = f'{file_name}.pdf'
    dst = f'html_pdf/{src}'
    os.rename(src, dst)
    print('Files saved in "html_pdf" folder')
    
    # clear cell output after some time
    time.sleep(cleaning_delay)
    clear_output(wait=False)
    
def print_date(d):
    """
    Prints current date. To use at the end of notebooks:
    
    Based on:
    https://www.programiz.com/python-programming/datetime/current-datetime
    
    C: 2021.11.02 / M: 2021.11.02    
    """
    from datetime import date
    from datetime import datetime
    
    today = date.today()
    now = datetime.now()

    # dd/mm/YY
    d1 = today.strftime("%d/%m/%Y")
    if d==1: print(d1)

    # Textual month, day and year	
    d2 = today.strftime("%B %d, %Y")
    if d==2: print(d2)

    # mm/dd/y
    d3 = today.strftime("%m/%d/%y")
    if d==3: print(d3)

    # Month abbreviation, day and year	
    d4 = today.strftime("%d-%b-%Y")
    if d==4: print(d4)
    
    
    d5 = now.strftime("%d-%b-%Y %H:%M:%S")
    if d==5: print(d5)
