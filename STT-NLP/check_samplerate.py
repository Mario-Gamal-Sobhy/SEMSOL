import sounddevice as sd

def query_devices():
    """Queries and prints audio device information."""
    try:
        print("Querying audio devices...")
        devices = sd.query_devices()
        print(devices)
        default_input_device = sd.default.device[0]
        print(f"\nDefault input device index: {default_input_device}")
        device_info = sd.query_devices(default_input_device, 'input')
        print(f"Default input device info: {device_info}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    query_devices()

