"""SHERIFS
Seismic Hazard and Earthquake Rates In Fault Systems

Version 1.2

@author: Thomas Chartier
"""
import numpy as np
from fpdf import FPDF

def create_title_page(Run_name, pdf):
    if False:
        while True:
            i = 10
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 18)
    pdf.cell(60)
    pdf.cell(90, 10, ' ', 0, 2, 'C')
    pdf.cell(90, 10, ' ', 0, 2, 'C')
    pdf.cell(90, 10, ' ', 0, 2, 'C')
    pdf.cell(75, 10, Run_name, 0, 2, 'C')
    pdf.cell(90, 10, ' ', 0, 2, 'C')
    pdf.cell(90, 10, ' ', 0, 2, 'C')
    return pdf

def print_lt(logictree, pdf):
    if False:
        for i in range(10):
            print('nop')
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 8)
    pdf.cell(90, 10, ' ', 0, 2, 'C')
    for key in logictree:
        for str_i in logictree[key]:
            pdf.cell(75, 10, str_i, 0, 2, 'C')
        pdf.cell(90, 10, ' ', 0, 2, 'C')
        pdf.cell(90, 10, ' ', 0, 2, 'C')
    return pdf

def compare_mfd_subareas(Run_name, pdf):
    if False:
        while True:
            i = 10
    return pdf

def create(Run_name, logictree):
    if False:
        return 10
    pdf = FPDF()
    pdf = create_title_page(Run_name, pdf)
    pdf = print_lt(logictree, pdf)
    pdf = compare_mfd_subareas(Run_name, pdf)
    pdf.output(Run_name + '/report_' + Run_name + '.pdf', 'F')