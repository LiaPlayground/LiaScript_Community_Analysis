"""
LiaScript Paper Pipeline
Core modules for data loading, analysis orchestration, and paper generation.
"""

from .data_loader import LiaScriptDataLoader
from .analysis_runner import AnalysisRunner
from .paper_builder import PaperBuilder

__all__ = ['LiaScriptDataLoader', 'AnalysisRunner', 'PaperBuilder']
__version__ = '1.0.0'
