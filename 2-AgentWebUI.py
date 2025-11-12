import os
import subprocess
import dotenv

# Load environment variables
dotenv.load_dotenv()
Google_API_key = os.getenv('GOOGLE_API_KEY')

def launch_adk_web_ui():
    """Launch the ADK web UI for the agent"""
    
    # Set the API key as environment variable for the subprocess
    env = os.environ.copy()
    env['GOOGLE_API_KEY'] = Google_API_key
    
    print("🚀 Launching ADK Web UI...")
    print("📍 The web interface will open in your browser")
    print("📂 Using agent from: agent_config.py")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        # Option 1: Launch with specific agent file
        # subprocess.run(['adk', 'web', '--agent', 'agent_config.py:searching_agent'], env=env, check=True)
        
        # Option 2: Launch with auto-discovery (will find agents in current directory)
        subprocess.run(['adk', 'web'], env=env, check=True)
        
    except KeyboardInterrupt:
        print("\n\n✅ ADK Web UI stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching ADK Web UI: {e}")
        print("\nMake sure you have ADK installed and configured properly.")
    except FileNotFoundError:
        print("❌ 'adk' command not found.")
        print("\nPlease ensure google-adk is installed:")
        print("  pip install google-adk")
        print("\nOr use the terminal command directly:")
        print("  adk web")

if __name__ == "__main__":
    launch_adk_web_ui()
