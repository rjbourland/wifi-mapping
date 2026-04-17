import subprocess
import re
import time
import csv
from datetime import datetime

def get_wifi_rssi():
    """
    Captures nearby WiFi signal strengths using 'netsh wlan show networks mode=bssid'.
    Returns a list of dictionaries containing SSID, BSSID, and Signal strength.
    """
    try:
        # Execute netsh command to get BSSID and signal strength
        result = subprocess.run(['netsh', 'wlan', 'show', 'networks', 'mode=bssid'], 
                                capture_output=True, text=True, check=True)
        output = result.stdout
        
        networks = []
        current_net = None
        
        for line in output.splitlines():
            line = line.strip()
            if not line: continue
            
            # 1. Detect SSID (e.g., "SSID 1 : MyNetwork")
            if line.startswith("SSID"):
                if current_net:
                    # Push previous network if it had data
                    if current_net.get('bssid'):
                        networks.append(current_net)
                
                parts = line.split(':', 1)
                ssid = parts[1].strip() if len(parts) > 1 else "Unknown"
                current_net = {'ssid': ssid, 'bssid': None, 'signal': None}
            
            elif current_net is not None:
                # 2. Detect Primary BSSID (e.g., "BSSID 1 : 80:cc:...")
                # We check for "BSSID" and a colon, but NOT "Band:" (which is for colocated APs)
                if "BSSID" in line and ":" in line and "Band:" not in line:
                    bssid = line.split(':', 1)[1].strip()
                    
                    # If we already found a BSSID for this SSID, push the previous one
                    if current_net['bssid'] is not None:
                        networks.append(current_net)
                    
                    current_net['bssid'] = bssid
                
                # 3. Detect Signal (e.g., "Signal : 93%")
                elif "Signal" in line and ":" in line:
                    signal_str = line.split(':', 1)[1].strip()
                    match = re.search(r'(\d+)%', signal_str)
                    if match:
                        current_net['signal'] = int(match.group(1))
                
                # 4. Detect Colocated APs (e.g., "BSSID: 80:cc..., Band: 6 GHz...")
                elif "BSSID:" in line and "Band:" in line:
                    # Extract BSSID from "BSSID: 80:cc:..., Band: ..."
                    parts = line.split(',', 1)
                    bssid_part = parts[0].split(':', 1)[1].strip()
                    
                    # Colocated APs are added as separate entries immediately.
                    # Since netsh doesn't give a separate signal line for these,
                    # we use the signal of the primary AP as a baseline or 0.
                    networks.append({
                        'ssid': current_net['ssid'],
                        'bssid': bssid_part,
                        'signal': current_net['signal'] if current_net['signal'] is not None else 0
                    })
        
        # Final push for the last network found
        if current_net and current_net.get('bssid'):
            networks.append(current_net)
            
        return networks

    except subprocess.CalledProcessError as e:
        print(f"Error executing netsh: {e}")
        return []

def main():
    print("Starting WiFi RSSI Scan (Proof of Concept)...")
    print("Press Ctrl+C to stop.")
    
    filename = "wifi_scan_results.csv"
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'ssid', 'bssid', 'signal'])
        
        try:
            while True:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                scan_data = get_wifi_rssi()
                
                if not scan_data:
                    print("No networks found or error occurred.")
                else:
                    for net in scan_data:
                        ssid = net.get('ssid', 'Unknown')
                        bssid = net.get('bssid', 'Unknown')
                        signal = net.get('signal', 0)
                        writer.writerow([timestamp, ssid, bssid, signal])
                        print(f"[{timestamp}] SSID: {ssid} | BSSID: {bssid} | Signal: {signal}%")
                
                f.flush()
                time.sleep(5) # Scan every 5 seconds
        except KeyboardInterrupt:
            print("\nScan stopped by user.")

if __name__ == "__main__":
    main()
