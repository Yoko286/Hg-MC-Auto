import os
import time
import pyautogui
import pygetwindow as gw
from datetime import datetime
import pyperclip

# Global coordinate configuration
COORD = {
    "file": (42, 31),
    "open": (92, 52),
    "open_filename": (982, 648),
    "data_right": (1153, 321),
    "export": (1217, 501),
    "save_filename": (872, 759)
}

CONFIG_FILE = "mouse_coordinates.config"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def get_user_input(prompt):
    return input(prompt)

def save_coordinates_to_file():
    """Save coordinates to configuration file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            for key, value in COORD.items():
                f.write(f"{key}:{value[0]},{value[1]}\n")
        log("Coordinates saved to configuration file")
        return True
    except Exception as e:
        log(f"Error saving coordinates: {str(e)}")
        return False

def load_coordinates_from_file():
    """Load coordinates from configuration file"""
    global COORD
    try:
        if not os.path.exists(CONFIG_FILE):
            return False
            
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, values = line.split(':', 1)
                    if ',' in values:
                        x, y = values.split(',')
                        COORD[key] = (int(x), int(y))
        log("Coordinates loaded from configuration file")
        return True
    except Exception as e:
        log(f"Error loading coordinates: {str(e)}")
        return False

def capture_mouse_position(point_name):
    """Capture mouse position with countdown"""
    log(f"Please move your mouse to the position for '{point_name}'")
    log("Starting countdown: 8 seconds...")
    
    # Simple countdown
    for i in range(8, 0, -1):
        print(f"Countdown: {i} seconds...")
        time.sleep(1)
    
    # Get mouse position directly
    try:
        x, y = pyautogui.position()
        log(f"Captured position for '{point_name}': ({x}, {y})")
        return (x, y)
    except Exception as e:
        log(f"Error capturing mouse position: {str(e)}")
        return None

def configure_coordinates():
    """Interactive coordinate configuration"""
    global COORD
    
    log("Starting coordinate configuration...")
    log("You will now configure 6 coordinate points for the automation, please maximize your Evaluation software interface!")
    
    points = [
        ("file", "File menu"),
        ("open", "Open Data File..."),
        ("open_filename", "Filename input box in open dialog"),
        ("data_right", "Right-click point in data area"),
        ("export", "ASCII Export"),
        ("save_filename", "Filename input box in save as dialog")
    ]
    
    for point_key, point_description in points:
        while True:
            input(f"Press Enter to capture position for: {point_description}...")
            
            position = capture_mouse_position(point_description)
            if position:
                COORD[point_key] = position
                log(f"Position for '{point_description}' set to: {position}")
                
                # Ask if user wants to retry this position
                retry = get_user_input("Press Enter to confirm this position, or type 'r' to retry: ")
                if retry.lower() != 'r':
                    break
                else:
                    log("Retrying position capture...")
            else:
                log("Failed to capture position. Please try again.")
    
    # Save coordinates to file
    if save_coordinates_to_file():
        log("All coordinates configured and saved successfully!")
    else:
        log("Coordinates configured but failed to save to file.")
    
    return True

def coordinate_selection_menu():
    """Coordinate configuration menu"""
    log("=== Coordinate Configuration ===")
    log("1. Configure coordinate positions")
    log("2. Use previously configured coordinates")
    
    while True:
        choice = get_user_input("Please select option (1 or 2): ")
        if choice == '1':
            return configure_coordinates()
        elif choice == '2':
            if load_coordinates_from_file():
                log("Using previously configured coordinates")
                return True
            else:
                log("No configuration file found. Please configure coordinates first.")
                return configure_coordinates()
        else:
            log("Invalid choice. Please enter 1 or 2.")

def activate_software():
    """Activate software window and ensure it's in foreground"""
    try:
        log("Activating software window...")
        wins = gw.getWindowsWithTitle("Data Evaluation")
        if not wins:
            raise RuntimeError("No window found with title containing 'Data Evaluation'!")
        
        win = wins[0]
        log(f"Found window: {win.title}")
        
        win.activate()
        time.sleep(0.5)
        win.maximize()
        time.sleep(0.5)
        
        log("Software window activated and maximized")
        return True
        
    except Exception as e:
        log(f"Failed to activate software: {str(e)}")
        return False

def wait_for_user_confirm():
    """Wait for user confirmation"""
    try:
        input("Press Enter to continue (or Ctrl+C to cancel)...")
        return True
    except KeyboardInterrupt:
        log("User cancelled operation")
        return False

def export_single_dat(filename, DAT_DIR, full_path):
    base, _ = os.path.splitext(filename)
    log(f"Processing → {filename}")

    try:
        pyautogui.click(*COORD["file"])
        time.sleep(0.3)
        
        pyautogui.click(*COORD["open"])
        time.sleep(0.2)
        
        time.sleep(0.3)
        
        pyautogui.click(*COORD["open_filename"])
        time.sleep(0.3)
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.2)
        pyperclip.copy(os.path.join(DAT_DIR, filename))
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.3)
        pyautogui.press('enter')
        time.sleep(0.5)
        
        pyautogui.rightClick(*COORD["data_right"])
        time.sleep(0.5)
        
        pyautogui.click(*COORD["export"])
        time.sleep(0.3)
        
        time.sleep(0.5)
        
        pyautogui.click(*COORD["save_filename"])
        time.sleep(0.3)
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.2)
        
        save_path = os.path.join(full_path, base + '.csv')
        pyperclip.copy(save_path)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.3)
        
        pyautogui.press('enter')
        time.sleep(0.4)
        
        log(f"✅ Exported → {base}.csv")
        return True
        
    except Exception as e:
        log(f"❌ Error processing {filename}: {str(e)}")
        for _ in range(3):
            pyautogui.press('esc')
            time.sleep(0.5)
        return False

def main():
    # Coordinate configuration
    if not coordinate_selection_menu():
        log("Coordinate configuration failed. Exiting.")
        return
    
    # Get user input for paths
    DAT_DIR = get_user_input("Please enter .dat file path: ")
    if not os.path.exists(DAT_DIR):
        log(f"Error: Directory does not exist {DAT_DIR}")
        return
    
    full_path = get_user_input("Please enter output file path: ")
    if not os.path.exists(os.path.dirname(full_path)):
        log(f"Error: Directory does not exist {os.path.dirname(full_path)}")
        return
    
    if not os.path.exists(DAT_DIR):
        log(f"Error: Directory does not exist {DAT_DIR}")
        return
    
    os.chdir(DAT_DIR)
    dat_files = [f for f in os.listdir() if f.lower().endswith('.dat')]
    
    if not dat_files:
        log("No .dat files found in directory")
        return
    
    log(f"Found {len(dat_files)} .dat files in total")
    
    for i, f in enumerate(dat_files, 1):
        log(f"{i}. {f}")
    
    # Final confirmation before starting automation
    log("All configurations completed. Ready to start automation.")
    if not wait_for_user_confirm():
        return
    
    if not activate_software():
        log("⚠️  Unable to automatically activate software window, please manually bring the software window to foreground")
        time.sleep(3)
    
    successful = 0
    failed = 0
    
    for idx, dat in enumerate(dat_files, 1):
        log(f"===== File {idx}/{len(dat_files)} =====")
        if export_single_dat(dat, DAT_DIR, full_path):
            successful += 1
        else:
            failed += 1
        time.sleep(1)
        
        if idx % 3 == 0 and idx < len(dat_files):
            log(f"Processed {idx} files, continuing with remaining {len(dat_files)-idx}...")
            time.sleep(1)
    
    log(f"Conversion completed! Successful: {successful}, Failed: {failed}")
    
    pyautogui.press('esc')
    time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Program interrupted by user")
    except Exception as e:
        log(f"Program runtime error: {str(e)}")
    finally:
        log("Program ended")