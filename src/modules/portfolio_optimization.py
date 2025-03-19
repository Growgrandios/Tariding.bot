# portfolio_optimization.py

import numpy as np
import pandas as pd
import scipy.optimize as sco
from datetime import datetime, timedelta
import logging
import json
import os
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any, Optional, Union, Tuple

# Konfiguration des Loggings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/portfolio_optimization.log"),
        logging.StreamHandler()
    ]
)

class PortfolioOptimizer:
    """
    Optimiert die Portfolioallokation basierend auf modernen Portfoliotheorien
    und Risiko-Rendite-Verhältnissen.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den PortfolioOptimizer.
        
        Args:
            config: Konfigurationseinstellungen
        """
        self.logger = logging.getLogger("PortfolioOptimizer")
        self.logger.info("Initialisiere PortfolioOptimizer...")
        
        # Konfiguration laden
        self.config = config or {}
        
        # Optimierungskonfiguration
