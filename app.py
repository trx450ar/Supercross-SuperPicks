<style>
    .stApp {
        background-color: #000000;
        color: #E0E0E0;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        color: #A67C52; /* Burnt bronze header accent */
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stButton > button {
        background-color: #2A2A2A;
        color: #E0E0E0;
        border: 1px solid #4A4A4A;
        border-radius: 4px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #3A3A3A;
        border-color: #B22222; /* Subtle red hover accent */
        box-shadow: 0 0 8px rgba(178, 34, 34, 0.2);
    }
    .stSidebar {
        background-color: #1A1A1A;
        border-right: 1px solid #3A3A3A;
    }
    .stExpander {
        background-color: #222222;
        border: 1px solid #3A3A3A;
        border-radius: 4px;
    }
    hr {
        border-color: #3A3A3A;
    }
    footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #0F0F0F;
        color: #666666;
        text-align: center;
        padding: 8px 0;
        font-size: 0.75em;
        border-top: 1px solid #B22222; /* Very thin red footer line */
    }
    /* Subtle geometrical/stealth touch - faint angular divider lines */
    .stDivider {
        background: linear-gradient(to right, transparent, #4A4A4A, transparent);
        height: 1px;
    }
</style>
