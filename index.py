import os
import sys

# Point Vercel to run app.py via Streamlit CLI
sys.argv = ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
from streamlit.web.cli import main

if __name__ == "__main__":
    main()
