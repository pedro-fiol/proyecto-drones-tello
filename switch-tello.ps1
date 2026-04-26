param([string]$mode = "tello")  # "tello" or "lan"

if ($mode -eq "tello") {
    Write-Host "Switching to Tello WiFi..." -ForegroundColor Yellow
    netsh interface set interface "Ethernet" admin=disable
    netsh wlan connect name="RMTT-AD3294"   # <-- replace with your Tello SSID
    Start-Sleep 3
    ping -n 2 192.168.10.1
    Write-Host "Done. Tello at 192.168.10.1" -ForegroundColor Green
} else {
    Write-Host "Switching to LAN..." -ForegroundColor Yellow
    netsh wlan disconnect
    netsh interface set interface "Ethernet" admin=enable
    Start-Sleep 2
    ping -n 2 8.8.8.8
    Write-Host "Done. LAN restored." -ForegroundColor Green
}
