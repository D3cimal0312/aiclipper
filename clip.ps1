# One-command launcher for the local AI clipper.
# Usage (from PowerShell):
#   .\clip.ps1 "C:\path\to\video.mp4"
#   .\clip.ps1 "C:\path\to\video.mp4" -Whisper small -Clips 8
#   .\clip.ps1 "C:\path\to\video.mp4" -Language ne
#   .\clip.ps1 "C:\path\to\video.mp4" -Clips 2 -Min 15 -Max 45 -Llm qwen2.5:7b
#   .\clip.ps1 "C:\path\to\video.mp4" -PostTimes "10:00,15:00,21:00" -StartDate 2026-11-20
#   .\clip.ps1 "C:\yt\gta6.mp4" -Mode gameplay -Note "police chase in Vice City" -Skip "2:10-5:40"
#   .\clip.ps1 "C:\yt\kai_stream.mp4" -Campaign kai            (Kai Cenat GTA 6 campaign)
#   .\clip.ps1 "C:\yt\kai_stream.mp4" -Campaign kai -Facecam top-right -Prefilter 12
param(
    [Parameter(Mandatory = $true)][string]$Video,
    [string]$Whisper = "medium",
    [int]$Clips = 6,
    [int]$Min = 10,
    [int]$Max = 20,
    [string]$Llm = "qwen2.5:14b",
    [string]$Language = "",
    [string]$CaptionStyle = "pop",
    [string]$CaptionColor = "yellow",
    [int]$CaptionWords = 0,
    [int]$CaptionSize = 0,
    [string]$Font = "",
    [switch]$NoCaptions,
    [string]$TitleLang = "English",
    [switch]$NoTitle,
    [double]$TitleSeconds = 0,
    [ValidateSet("speech", "gameplay")][string]$Mode = "speech",
    [switch]$Commentary,
    [ValidateSet("", "crop", "blur", "stack")][string]$Layout = "",
    [string]$Campaign = "",
    [string]$Facecam = "",
    [int]$Prefilter = 0,
    [string]$AudienceOffset = "",
    [string]$LocalOffset = "",
    [string]$Note = "",
    [string]$Skip = "",
    [string]$Vision = "",
    [string]$ChannelFile = "",
    [switch]$NoThumbs,
    [string]$ThumbVision = "",
    [string]$PostTimes = "09:00,14:00,20:00",
    [string]$StartDate = ""
)

# Resolve the video path before changing folders
$Video = (Resolve-Path $Video).Path

# Work from the folder this script lives in (C:\clipper)
Set-Location $PSScriptRoot

# Turn on the Python environment
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# Tell Windows where the NVIDIA libraries for Whisper are
$sp = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
$env:PATH = "$sp\nvidia\cublas\bin;$sp\nvidia\cudnn\bin;" + $env:PATH

# Hide a harmless Hugging Face warning on Windows
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# Build and run the clipper command
$argsList = @(
    $Video,
    "--device", "cuda",
    "--compute-type", "float16",
    "--whisper", $Whisper,
    "--llm", $Llm
)
# With -Campaign the profile (channel_kai.json) supplies clips/min/max/layout/post times unless you pass them yourself
$BoundParams = $PSBoundParameters
function Use($n) { return ($Campaign -eq "" -or $script:BoundParams.ContainsKey($n)) }
if (Use "Clips") { $argsList += @("--clips", $Clips) }
if (Use "Min") { $argsList += @("--min", $Min) }
if (Use "Max") { $argsList += @("--max", $Max) }
if ($Language -ne "") { $argsList += @("--language", $Language) }
$argsList += @("--caption-style", $CaptionStyle, "--caption-color", $CaptionColor)
if ($CaptionWords -gt 0) { $argsList += @("--caption-words", $CaptionWords) }
if ($CaptionSize -gt 0) { $argsList += @("--caption-size", $CaptionSize) }
if ($Font -ne "") { $argsList += @("--font", $Font) }
if ($NoCaptions) { $argsList += "--no-captions" }
if ($NoTitle) { $argsList += "--no-title" }
$argsList += @("--title-lang", $TitleLang)
if (Use "TitleSeconds") { $argsList += @("--title-seconds", $TitleSeconds) }

# Channel settings (show name, tags, hashtag, mandatory line): channel.json next to this script
if ($ChannelFile -eq "" -and $Campaign -ne "") {
    $cf = "$PSScriptRoot\channel_$Campaign.json"
    if (-not (Test-Path $cf)) { Write-Error "Campaign file not found: $cf"; exit 1 }
    $ChannelFile = $cf
}
if ($ChannelFile -eq "" -and $Mode -eq "gameplay" -and (Test-Path "$PSScriptRoot\channel_gta6.json")) { $ChannelFile = "$PSScriptRoot\channel_gta6.json" }
if ($ChannelFile -eq "" -and (Test-Path "$PSScriptRoot\channel.json")) { $ChannelFile = "$PSScriptRoot\channel.json" }

# Gameplay mode (GTA VI etc.): pick action moments from audio + motion instead of speech
$argsList += @("--mode", $Mode)
if ($Commentary) { $argsList += "--commentary" }
if ($Layout -ne "") { $argsList += @("--layout", $Layout) }
if ($Facecam -ne "") { $argsList += @("--facecam", $Facecam) }
if ($Prefilter -gt 0) { $argsList += @("--prefilter", $Prefilter) }
if ($AudienceOffset -ne "") { $argsList += @("--audience-offset", $AudienceOffset) }
if ($LocalOffset -ne "") { $argsList += @("--local-offset", $LocalOffset) }
if ($Note -ne "") { $argsList += @("--clip-note", $Note) }
if ($Skip -ne "") { $argsList += @("--skip", $Skip) }
if ($Vision -ne "") { $argsList += @("--vision", $Vision) }
if ($ChannelFile -ne "") { $argsList += @("--profile", $ChannelFile) }

# Thumbnails and posting plan
if ($NoThumbs) { $argsList += "--no-thumbs" }
if ($ThumbVision -ne "") { $argsList += @("--thumb-vision", $ThumbVision) }
if (Use "PostTimes") { $argsList += @("--post-times", $PostTimes) }
if ($StartDate -ne "") { $argsList += @("--start-date", $StartDate) }

python clipper.py @argsList