#!/usr/bin/env python3
"""
WordWizard Setup Script
Automated setup for the WordWizard AI Text Prediction application
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {python_version.major}.{python_version.minor}")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor} is compatible")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = os.path.join("mysite", "venv")
    
    if not os.path.exists(venv_path):
        print("🔄 Creating virtual environment...")
        if not run_command(f"cd mysite && python -m venv venv", "Virtual environment creation"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Activation command depends on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
    else:
        activate_cmd = "source venv/bin/activate"
    
    print(f"💡 To activate virtual environment, run: cd mysite && {activate_cmd}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    os.chdir("mysite")
    
    # Try different python commands
    python_commands = ["python", "python3", "py"]
    pip_commands = ["pip", "pip3"]
    
    for python_cmd in python_commands:
        try:
            subprocess.run([python_cmd, "--version"], check=True, capture_output=True)
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    else:
        print("❌ No Python interpreter found")
        return False
    
    for pip_cmd in pip_commands:
        if run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
            return True
    
    print("❌ Failed to install dependencies with pip")
    return False

def setup_django():
    """Setup Django application"""
    commands = [
        ("python manage.py migrate", "Running database migrations"),
        ("python manage.py collectstatic --noinput", "Collecting static files"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  {description} failed, but continuing...")
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("🔄 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# WordWizard Environment Variables\n")
            f.write("Django_Key=django-insecure-change-me-in-production\n")
            f.write("DEBUG=True\n")
            f.write("# DNS_address=your-domain.com\n")
            f.write("# IP_address=your-ip-address\n")
        print("✅ .env file created")
        print("💡 Please update the Django_Key in .env file for production")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🧙‍♂️ WordWizard Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to setup virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Setup Django
    if not setup_django():
        print("❌ Failed to setup Django")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 WordWizard setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. cd mysite")
    if platform.system() == "Windows":
        print("2. venv\\Scripts\\activate")
    else:
        print("2. source venv/bin/activate")
    print("3. python manage.py runserver")
    print("4. Open http://127.0.0.1:8000/ in your browser")
    print("\n💡 Don't forget to update the Django_Key in .env for production!")

if __name__ == "__main__":
    main()