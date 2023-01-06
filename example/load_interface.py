import sys
sys.path.extend(['../src'])

from IPython.display import display,clear_output
from parameter_UI import my_UI

wd_hydroroot = my_UI()
wd_hydroroot.display()
