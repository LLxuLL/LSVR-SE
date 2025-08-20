YOUR_PROJECT_PATH="/path/to/your/project"

if [ -f "$YOUR_PROJECT_PATH/LSVR-SE/.venv/bin/activate" ]; then
    source "$YOUR_PROJECT_PATH/LSVR-SE/.venv/bin/activate"
fi

"$YOUR_PROJECT_PATH/LSVR-SE/.venv/bin/python" -m streamlit run "$YOUR_PROJECT_PATH/LSVR-SE/application.py"

read -p "Press enter to continue..."