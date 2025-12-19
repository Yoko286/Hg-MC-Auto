import os
import subprocess
import sys

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the program header"""
    print("=" * 60)
    print("          Welcome to Hg_MC_Auto!")
    print("=" * 60)

def print_menu():
    """Print the main menu options"""
    print("\nPlease select a task:")
    print("1. Automatically export isotope data")
    print("2. Automatically export instrument parameters, merge isotope data, and calculate isotope fractionation values")
    print("3. Classify data using an empirical model")
    print("4. Classify data using a machine learning model")
    print("5. Train your own machine learning expert model")
    print("0. Exit")
    print("-" * 60)

def get_script_path(script_name):
    """Get the full path of a script in the same directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    return script_path

def run_script(script_name):
    """Run a Python script in the same directory"""
    script_path = get_script_path(script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script '{script_name}' not found in the current directory.")
        print(f"   Expected path: {script_path}")
        return False
    
    try:
        print(f"🚀 Running {script_name}...")
        print("-" * 40)
        
        # Run the script using the same Python interpreter
        result = subprocess.run([sys.executable, script_path], check=True)
        
        print("-" * 40)
        print(f"✅ {script_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Error: Python interpreter not found.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {script_name}: {e}")
        return False

def wait_for_enter():
    """Wait for user to press Enter"""
    input("\nPress Enter to continue...")

def main():
    """Main program loop"""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            print("\nThank you for using Hg_MC_Auto! Goodbye!")
            break
            
        elif choice == '1':
            print("\nYou selected: Automatically export isotope data")
            success = run_script("1_Automatic_export.py")
            wait_for_enter()
            
        elif choice == '2':
            print("\nYou selected: Automatically export instrument parameters, merge isotope data, and calculate isotope fractionation values")
            success = run_script("2_Automatic_calculation.py")
            wait_for_enter()
            
        elif choice == '3':
            print("\nYou selected: Classify data using an empirical model")
            success = run_script("3_Empirical_model.py")
            wait_for_enter()
            
        elif choice == '4':
            print("\nYou selected: Classify data using a machine learning model")
            success = run_script("4_ML_Predict.py")
            wait_for_enter()
            
        elif choice == '5':
            print("\nYou selected: Train your own machine learning expert model")
            print("This will run two scripts sequentially:")
            print("1. 5.Exter_ML_train.py")
            print("2. 6_Inter_ML_train.py")
            
            confirm = input("\nDo you want to proceed? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                # Run first script
                print("\n" + "="*50)
                print("Running 5.Exter_ML_train.py")
                print("="*50)
                success1 = run_script("5.Exter_ML_train.py")
                
                if success1:
                    print("\n" + "="*50)
                    print("Running 6_Inter_ML_train.py...")
                    print("="*50)
                    success2 = run_script("6_Inter_ML_train.py")
                    
                    if success2:
                        print("\n🎉 Machine learning training completed successfully!")
                    else:
                        print("\n⚠️ Machine learning training had some issues.")
                else:
                    print("\n❌ Cannot proceed with training - prediction script failed.")
            else:
                print("Operation cancelled.")
                
            wait_for_enter()
            
        else:
            print("\n❌ Invalid choice. Please enter a number between 0 and 5.")
            wait_for_enter()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        print("Please check your installation and try again.")