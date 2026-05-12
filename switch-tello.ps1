param([string]$mode = "tello")  # "tello", "lan", or "dual"

if ($mode -eq "tello") {
    Write-Host "Switching to Tello WiFi (Outdoor/Solo Tello mode)..." -ForegroundColor Yellow
    netsh interface set interface "Ethernet" admin=disable
    netsh wlan connect name="RMTT-AD3294"   # <-- replace with your Tello SSID
    Start-Sleep 3
    ping -n 2 192.168.10.1
    Write-Host "Done. Tello at 192.168.10.1" -ForegroundColor Green
} elseif ($mode -eq "dual") {
    Write-Host "Connecting to Tello WiFi while keeping LAN (Dev mode)..." -ForegroundColor Yellow
    netsh interface set interface "Ethernet" admin=enable
    netsh wlan connect name="RMTT-AD3294"
    Start-Sleep 3
    ping -n 2 192.168.10.1
    Write-Host "Done. Tello at 192.168.10.1 AND Ethernet enabled." -ForegroundColor Green
} else {
    Write-Host "Switching to LAN..." -ForegroundColor Yellow
    netsh wlan disconnect
    netsh interface set interface "Ethernet" admin=enable
    Start-Sleep 2
    ping -n 2 8.8.8.8
    Write-Host "Done. LAN restored." -ForegroundColor Green
}
