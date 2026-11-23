# How to Use the Local AI Clipper (v4: YouTube Shorts edition)

Day-to-day guide for your Windows laptop (RTX 4050, 6 GB). Setup is finished: Python 3.11 environment, ffmpeg, Ollama with `qwen2.5:7b`, and the packages in `C:\clipper\venv`.

```
C:\clipper\
    clipper.py      <- the AI clipper
    clip.ps1        <- launcher (use this every time)
    channel.json    <- YOUR channel settings (show name, tags, hashtag, mandatory line)
    channel_gta6.json <- same, for GTA VI gameplay (used automatically with -Mode gameplay)
    venv\           <- Python environment (don't touch)
    clips\          <- results appear here
```

---

## One-time setup for this version

1. **Download the 14B model** (about 9 GB, once): `ollama pull qwen2.5:14b`
2. **Install Pillow** (needed for thumbnails). In PowerShell:
   ```
   cd C:\clipper
   venv\Scripts\python -m pip install pillow
   ```
3. **Put `channel.json` in `C:\clipper`** and open it in Notepad. Replace `PASTE YOUR EXACT MANDATORY LINE HERE` with your exact required line. Until you do, every description shows `[[MANDATORY LINE MISSING...]]` and the script warns you.
4. Replace `clipper.py` and `clip.ps1` with the new files, then run `Unblock-File C:\clipper\clip.ps1`.

---

## The 3 rules

1. **Always start from `C:\clipper`** (`cd C:\clipper`).
2. **Use the launcher `.\clip.ps1`.** It turns on the right Python and the GPU libraries. If you run `python clipper.py` yourself, the prompt must start with `(venv)`.
3. **Ollama must be running** (icon in the system tray, or open it from the Start menu).

---

## Normal use

```
cd C:\clipper
.\clip.ps1 "C:\Users\Downloads\video.mov"
```

Tip: type `.\clip.ps1 ` then drag the video into the PowerShell window to paste its path.

**Defaults now follow the YouTube advice:** clips are **10 to 20 seconds**, trimmed so they start on the first spoken word and end right after the last one, the hook title shows in the first second and stays for the whole clip, and the AI looks for a punchline at the end so clips loop cleanly.

### What you get per video

In `C:\clipper\clips\<video name>\`:

| File | What it is |
|---|---|
| `clip_01.mp4` ... | Finished vertical clips (hook title + captions) |
| `thumb_01.jpg` ... | Thumbnail: best face frame, hook text, key word in colour |
| `thumb_01_clean.jpg` ... | Same frame without text (use if you want your own text) |
| `upload_plan.md` | **Open this to post.** For each clip: publish time, YouTube title, description, tags, thumbnail to use. Copy and paste. |
| `clips.json` | All the data (scores, titles, durations, best-frame time) |
| `transcript.json` | Full transcript |
| `clip_01.ass` ... | Caption files (safe to delete) |

---

## GTA VI and other gameplay: gameplay mode

Speech mode picks clips from what people say. Gameplay is mostly silent, so **gameplay mode** picks the biggest *action* moments instead.

```
cd C:\clipper
.\clip.ps1 "C:\yt\gta6_session.mp4" -Mode gameplay -Clips 6 -Note "police chase in Vice City" -Skip "2:10-5:40"
```

`-Mode gameplay` automatically uses `channel_gta6.json` (show "GTA VI", characters Lucia and Jason, tags for Vice City and Leonida). Open it in Notepad and fill in `mandatory_line` like you did for `channel.json`.

| Step | What happens |
|---|---|
| 1. Find action | Every half second the tool measures game-audio loudness, sudden loudness jumps (crashes, explosions, gunshots) and how fast the picture changes. High on all three = action. |
| 2. Cut clips | Takes the biggest peaks, at least one clip length apart. Each clip starts about 9 seconds before the peak so the build-up is included. Length is the middle of `-Min`/`-Max` (15 s by default). If fewer clear moments exist than you asked for, you get fewer clips, not weak ones. |
| 3. Layout | The whole game screen (HUD included) sits in the middle on a blurred copy of itself, with the hook title above. Use `-Layout crop` for a tight centre crop instead. |
| 4. Title | The AI writes it from what you tell it (`-Note`) and, if you use `-Vision`, short descriptions of 3 frames per clip. It is told not to invent events, names or story. With neither, titles stay generic ("wild moment") because the AI cannot see the video. |
| 5. Thumbnail | The sharpest, most colourful frame near the action peak, full frame on a blurred background with the hook text above it. No faces needed. |

| Gameplay option | What it does | Example |
|---|---|---|
| `-Mode gameplay` | Turns this mode on | `-Mode gameplay` |
| `-Note` | One line about what the video shows. The AI uses it for titles. | `-Note "chasing a boat through Vice City"` |
| `-Skip` | Time ranges to leave out entirely, `start-end`, several separated by commas, `m:ss` or `h:mm:ss`. **Use it for story cutscenes** (see rules below). | `-Skip "2:10-5:40,1:02:00-1:05:30"` |
| `-Vision` | Optional local vision model that describes frames and rates thumbnails. Download first, for example `ollama pull qwen3-vl:2b`. | `-Vision qwen3-vl:2b` |
| `-Commentary` | You talk over the game: also transcribe your voice for captions and title context (slower, uses Whisper). | `-Commentary` |
| `-Layout` | `blur` (default in gameplay mode) or `crop` | `-Layout crop` |

Limits you should know about: it measures loudness and motion, not meaning. A loud music swell or a fast camera pan can score high, and a quiet stealth takedown can score low. It also cannot tell a cutscene from gameplay, which is why `-Skip` exists. Always watch the clips before posting.

### Rockstar's rules for posting GTA footage (read the original)

This summary comes from [Rockstar's policy page](https://support.rockstargames.com/articles/200153756/rockstar-games-video-policy) as I read it, plus a [November 2026 article](https://tbreak.com/gta-6-spoilers-rockstar-video-takedown-policy/). I am not a lawyer, so read the original yourself.

- **Non-commercial fan use.** Gameplay clips are tolerated for non-commercial use. As summarized, monetization and ads are **not** permitted. If you plan to earn from these Shorts, ask Take-Two first at the email on that page (copyright@take2games.com) before relying on this tool's growth plan.
- **Spoilers get taken down:** endings, big story reveals, edited cutscene or dialogue compilations and standalone cinematics. Cutscenes inside a longer playthrough are fine. Mark story sections with `-Skip`.
- **Leaked or pre-release footage is not allowed**, however you got it.
- **No cheating or mod-menu content**, and no in-game TV or comedy shown in isolation.
- Only clip footage **you recorded yourself** (or have permission to use). Do not run other people's streams through the tool.

---

## Kai Cenat GTA 6 Marathon campaign

One command makes the Shorts, the TikTok/YouTube metadata and a rule check:

```
cd C:\clipper
.\clip.ps1 "C:\yt\kai_stream.mp4" -Campaign kai
```

`-Campaign kai` loads `channel_kai.json` (put it in `C:\clipper`, then `Unblock-File` the new `clip.ps1`/`clipper.py` as before). It sets: English only, 8 clips of 12-25 s, the facecam stack layout, and posting times for a US audience. Anything you type yourself still wins, for example `-Clips 5 -Max 30`.

### Get the footage right (your job, the tool cannot check this)

- Download only from **Kai Cenat Live, Kai Cenat Gaming or his Twitch**, from the marathon (Nov 18 onward). Cut from the full stream/VOD, never from his existing Shorts. The tool warns if your file is vertical or under 3 minutes.
- Use **1-2 hour chunks**, not a 10-hour VOD: `yt-dlp --download-sections "*1:00:00-2:30:00" URL`. For a long file add `-Prefilter 12` (only the 12 most hyped 8-minute windows go to the AI) and `-Llm qwen2.5:7b` to keep it fast.
- Use `-Skip "start-end"` for story cutscenes and any break/BRB part (Rockstar takes down spoilers).

### What happens to each clip

| Step | Result |
|---|---|
| Pick moments | The AI is told to find funny moments, quotable lines and big reactions, and to skip arguments, insults and non-English speech. |
| Layout `stack` | Gameplay on top, **his facecam big underneath**, blurred background. The tool finds the webcam corner by itself. Open `facecam_check.jpg` in the output folder: the green box must surround his camera. If not, use `-Facecam top-right` (or `bottom-left`, ...) or exact pixels `-Facecam 1268,0,652,452`. No facecam found: it falls back to `blur` and warns you. |
| On-screen text | `Kai Cenat: <hook title>` is shown the **whole clip** (`-NoTitle` is ignored for this campaign). Word captions are on as well. |
| YouTube title | `Hook 😂 | Kai Cenat` |
| Description | Summary starting with "Kai Cenat", source line, question, `#KAICENAT #GTA6 #Shorts` |
| TikTok caption | `Kai Cenat: hook 🔥 #KAICENAT #GTA6 #kaicenatclips #fyp`, always 150 characters or less |
| Thumbnail | Taken from the clip itself, with that clip's own hook text, so thumbnails never repeat or mismatch. |
| Schedule | Times are in US Eastern (UTC-5) at 12:00, 17:00 and 20:00. `upload_plan.md` shows **your Nepal time first** (for example 22:45, 03:45, 06:45), because YouTube and TikTok scheduling use your own clock. Change with `-AudienceOffset -8` (US Pacific), `-PostTimes "13:00,18:00"`. |

### Files per video (in `clips\<video name>\`)

| File | Use |
|---|---|
| `clip_01.mp4` ... | Upload these |
| `clip_01.json` ... | The short metadata file for each clip: `youtube` (title, description, tags, hashtags, category), `tiktok` (caption, hashtags), publish times, source timestamps, `checks` and `review_by_eye` |
| `thumb_01.jpg` ... | YouTube thumbnails (`_clean` = no text) |
| `upload_plan.md` | All of it in copy-paste order |
| `facecam_check.jpg` | Check the webcam box |

If the end of the run says "Self-check failed for ...", open that clip's JSON: `checks` shows which rule failed (for example Japanese text in the speech, or a banned word).

### Rule by rule

| Campaign rule | Who handles it |
|---|---|
| "Kai Cenat" in caption or overlay | Tool: overlay every clip + title + TikTok caption. Checked in `checks`. |
| `#KAICENAT` in captions | Tool: in the YouTube description and the TikTok caption. Checked. |
| Kai the main focus, in the edit throughout | Tool: stack layout keeps his facecam on screen, and `kai_visible_share` in each JSON is a rough face-detection hint. **You** watch each clip. |
| No negative portrayal, hate farming, false narratives | Tool: AI told to stay positive, words in `banned_words` (edit in `channel_kai.json`) blocked from titles, descriptions, tags. **You** judge the clip itself. |
| No Japanese captions or descriptions | Tool: English forced, Japanese characters filtered out and flagged. |
| No logos of any kind | Tool adds none. **You** check that the stream picture has no channel logo/watermark in your crop. If his facecam area has one, pick `-Layout crop`. |
| No repetitive or mismatched thumbnails | Tool: one thumbnail per clip from its own frame, unique titles checked. |
| No reposting his Shorts, only allowed channels, dates | **You.** |
| No view manipulation | **You** (no bots, no buying views). |
| 50%+ audience in US/CA/AU/UK/NZ, niche not vastly different | **You**, on the account. Post in those time zones, keep the account on gaming/streamer clips. |
| Allowed platforms | TikTok, Reels, Shorts, X. Use the same file. |

Rockstar's non-commercial rule (above) still applies to the GTA footage inside his stream.

---

## Your posting routine (from your YouTube advice)

| Advice | How the tool helps |
|---|---|
| Keep people watching: hook in the first second, 10-20 s, no slow lead-in, punchline at the end | Defaults are 10-20 s, edges are trimmed to speech, the AI is told to start on the hook and end on the punchline, and the hook title is on screen from 0:00. |
| Post 2-3 a day at the same times for 2+ weeks | `upload_plan.md` gives every clip a publish time. Default times are 09:00, 14:00 and 20:00 (your computer's time), starting tomorrow. Best-scoring clips go first. Change with `-PostTimes` and `-StartDate`. |
| One kind of content | `channel.json` keeps the show name, tags and hashtag the same on every upload. |
| Curiosity titles, hook and character first, show name last, one emoji, key words in the first 40 characters | Format: `Hook words 💀 | Scrubs`. The hook is capped at about 55 characters and the AI is told to put the character and hook words first. |
| Description: character names, Sacred Heart, Hulu, ABC, a question, the mandatory line, only `#ScrubsPartner` | Built automatically in that order. Hashtags the AI invents are stripped, so only the one in `channel.json` appears. |
| Hidden tags | Tags line in `upload_plan.md` (show tags + characters + AI search phrases), kept under the 500-character limit. |
| Best thumbnail frame: clear face, funny expression, no cut subtitle | The tool scans each clip, scores faces (size, eyes open, smile, sharpness) and builds `thumb_XX.jpg`. It also lists the best frame's time so you can pick that moment in Studio. |

Batch tip: use a long source video with `-Clips 8` to get a week of Shorts in one run.

---

## All launcher options (`.\clip.ps1`)

Put these after the video path.

| Option | What it does | Default | Example |
|---|---|---|---|
| `"video path"` | The video to clip (required, first) | none | `.\clip.ps1 "C:\Videos\a.mov"` |
| `-Clips` | How many clips to make | `6` | `-Clips 8` |
| `-Min` / `-Max` | Shortest / longest clip in seconds | `10` / `20` | `-Min 15 -Max 30` |
| `-Whisper` | Transcription model: `tiny`, `base`, `small`, `medium`, `large-v3` | `medium` | `-Whisper small` |
| `-Language` | Spoken language code (`en`, `ne`, `hi`). Blank = auto-detect. | auto | `-Language en` |
| `-Llm` | Ollama model that picks moments and writes titles | `qwen2.5:14b` | `-Llm qwen2.5:7b` (faster, fully on the GPU) |
| `-Mode` | `speech` (pick clips from what is said) or `gameplay` (pick action moments, see the GTA VI section) | `speech` | `-Mode gameplay` |
| `-Note` / `-Skip` / `-Vision` / `-Commentary` / `-Layout` | Gameplay options, explained in the GTA VI section above | off | `-Note "boat chase"` |
| `-ChannelFile` | Path to a channel settings file | `channel_gta6.json` in gameplay mode, otherwise `channel.json` (if present in `C:\clipper`) | `-ChannelFile "C:\x\other.json"` |
| `-TitleLang` | Language of the AI titles (keep English unless your font supports the script) | `English` | `-TitleLang English` |
| `-NoTitle` | No hook title on the video (metadata and thumbnails still get titles) | title on | `-NoTitle` |
| `-TitleSeconds` | Seconds the on-video title shows. `0` = whole clip. | `0` | `-TitleSeconds 3` |
| `-CaptionStyle` | `pop` (big, current word pops), `classic`, `clean` (whole phrases, no highlight) | `pop` | `-CaptionStyle clean` |
| `-CaptionColor` | Highlight colour for captions and thumbnail key word: `yellow`, `green`, `cyan`, `pink`, `orange`, `red`, `magenta`, `white`, or hex `#FFD400` | `yellow` | `-CaptionColor orange` |
| `-CaptionWords` | Words shown at once (0 = style default) | style default | `-CaptionWords 2` |
| `-CaptionSize` | Caption font size (0 = style default) | style default | `-CaptionSize 100` |
| `-Font` | Caption font name (must be installed) | `Arial Black` for pop | `-Font "Impact"` |
| `-NoCaptions` | Don't burn captions into the video | captions on | `-NoCaptions` |
| `-NoThumbs` | Don't make thumbnails | thumbnails on | `-NoThumbs` |
| `-ThumbVision` | Optional local vision model that helps pick the best thumbnail frame (see below) | off | `-ThumbVision qwen3-vl:2b` |
| `-PostTimes` | Daily upload times, `HH:MM`, comma separated. One Short per time. | `09:00,14:00,20:00` | `-PostTimes "10:00,16:00"` |
| `-StartDate` | First upload day, `YYYY-MM-DD` | tomorrow | `-StartDate 2026-11-20` |

### Handy combinations

| What you want | Command |
|---|---|
| Normal run | `.\clip.ps1 "video.mov"` |
| A week of Shorts from one episode | `.\clip.ps1 "video.mov" -Clips 8` |
| Quick test | `.\clip.ps1 "video.mov" -Clips 2 -Whisper small` |
| Slightly longer clips | `.\clip.ps1 "video.mov" -Min 15 -Max 30` |
| Calm captions, green highlight | `.\clip.ps1 "video.mov" -CaptionStyle clean -CaptionColor green` |
| No text on the video at all | `.\clip.ps1 "video.mov" -NoTitle -NoCaptions` |
| Clips only, no thumbnails | `.\clip.ps1 "video.mov" -NoThumbs` |
| Uploads at 10:00 and 18:00 from a set day | `.\clip.ps1 "video.mov" -PostTimes "10:00,18:00" -StartDate 2026-11-20` |

---

## channel.json (your channel settings)

| Field | What it is |
|---|---|
| `show` | Name added at the end of each YouTube title (`... \| Scrubs`) |
| `hashtag` | The only hashtag put in descriptions |
| `characters` | Names the AI may list as "Featuring". It only picks from this list, so it can't invent names. |
| `description_keywords` | A line placed in every description (your Sacred Heart / Hulu / ABC wording) |
| `base_tags` | Tags added to every upload |
| `mandatory_line` | Your exact required line. **Must be filled in by you.** |
| `emojis` | Emojis the AI may choose from (it uses one per title) |

Edit it in Notepad and keep the quotes and commas as they are. Save as UTF-8 (Notepad's default).

---

## Thumbnails

- The tool samples frames across each clip, scores them (face size, eyes open, smile, sharpness), then builds a vertical 1080x1920 image: face-centred, slightly boosted colour, big hook text with one word in your highlight colour.
- The text moves to the top or bottom, whichever leaves the face uncovered.
- `upload_plan.md` lists the best frame's time (for example "0:07 in the clip"). In Studio you can pick that moment from the frame picker, or upload `thumb_XX.jpg` as a custom thumbnail if your account offers it for Shorts.
- Face detection is basic (OpenCV). It can miss turned heads or pick a wrong face in a crowded scene, so glance at every thumbnail.

### Better AI models (optional)

| Use | Model | Notes |
|---|---|---|
| Titles and clip picking (default) | `qwen2.5:14b` | About 9 GB, so part of it runs from system RAM (needs roughly 16 GB RAM). Slower per request, better output. Download once with `ollama pull qwen2.5:14b`. `ollama ps` will show a CPU/GPU split, which is expected. |
| Faster fallback | `qwen2.5:7b` | Fits fully on the 6 GB GPU. Use `-Llm qwen2.5:7b` if the 14B is too slow or your RAM is low. |
| Alternative text model to try | `llama3.1:8b` | Run with `-Llm llama3.1:8b` and compare titles. |
| Smaller, newer options from a 6 GB VRAM guide (untested for this task) | `qwen3.5:4b`, `gemma4:e2b-it-qat` | Download with `ollama pull`, then try with `-Llm`. Use the `-it-qat` Gemma tag; the plain one needs more than 6 GB. |
| Vision judge for thumbnails | `qwen3-vl:2b` | `ollama pull qwen3-vl:2b`, then add `-ThumbVision qwen3-vl:2b`. It rates the 4 best frames per clip and adds to the score. Small vision models are rough judges, so compare its picks with the default. |

After loading any model, `ollama ps` should show `100% GPU`.

---

## Extra options (only with `python clipper.py`)

Every launcher option exists as a two-dash lower-case option (`--clips`, `--min`, `--max`, `--whisper`, `--language`, `--llm`, `--profile`, `--title-lang`, `--no-title`, `--title-seconds`, `--caption-style`, `--caption-color`, `--caption-words`, `--caption-size`, `--font`, `--no-captions`, `--no-thumbs`, `--thumb-vision`, `--post-times`, `--start-date`, `--mode`, `--commentary`, `--layout`, `--facecam`, `--prefilter`, `--audience-offset`, `--local-offset`, `--clip-note`, `--skip`, `--vision`). The launcher style (`-NoCaptions`) does not work with `python clipper.py`, and the other way round.

These work only with `python clipper.py`:

| Option | What it does | Default |
|---|---|---|
| `--device` | `cuda` (GPU), `cpu`, or `auto` | `auto` |
| `--compute-type` | Whisper precision: `float16`, `int8_float16`, `int8` | `auto` |
| `--window` | Seconds of transcript sent to the AI per request | `480` |
| `--out` | Folder for results | `clips` |
| `--thumb-font` | Thumbnail font file name, for example `impact.ttf` | Impact, then Arial Black |

Manual way (the prompt must show `(venv)`):

```
cd C:\clipper
venv\Scripts\activate
$sp = python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
$env:PATH = "$sp\nvidia\cublas\bin;$sp\nvidia\cudnn\bin;" + $env:PATH
python clipper.py "C:\path\to\video.mov" --device cuda --compute-type float16 --whisper medium --profile channel.json
```

---

## What you'll see while it runs

| Message | What's happening |
|---|---|
| `[1/4] Transcribing...` | Whisper converts speech to text. First run downloads the model. |
| `[2/4] Asking local LLM...` | The AI picks moments (one request per ~8 minutes of video). |
| `Writing hook titles, descriptions and tags...` | One short AI request per clip. |
| `[3/4] + [4/4] Rendering N clips and thumbnails...` | Each clip is cut, cropped, titled, captioned, and its thumbnail made. |

Low GPU use during downloads and rendering is normal (encoding runs on the CPU).

---

## Before you upload

1. Watch every clip and look at every thumbnail. The AI reads only the transcript and can't judge delivery.
2. Open `upload_plan.md`. Nothing needs filling in by hand unless you see `[[MANDATORY LINE MISSING...]]` (fix it in `channel.json` and run again).
3. Check that each title matches its clip. Edit freely.
4. Schedule in Studio at the times shown.

---

## Troubleshooting

| Message | Fix |
|---|---|
| `Thumbnails need Pillow` | Run `cd C:\clipper` then `venv\Scripts\python -m pip install pillow`, or add `-NoThumbs`. |
| `WARNING: no mandatory line set` | Edit `channel.json` and replace the placeholder with your exact line. |
| `Only N clips fit your length range` | The video doesn't have enough moments of 10-20 s. Widen with `-Min 8 -Max 30`, or use a longer video. |
| `No clear action moments found` / `Only N clear action moments found` | Gameplay mode only keeps real peaks. Use a longer video, lower `-Min`, or check the game audio is not muted in the recording. |
| Gameplay clips have empty blurred bands above and below | That is the `blur` layout. Use `-Layout crop` for a tighter centre crop. |
| `Could not read channel file` | `channel.json` has a typo (missing quote or comma). Compare with the original. |
| `Bad time ... in --post-times` | Use `HH:MM` like `09:00,14:00,20:00` inside quotes. |
| `face detector data not found, using a centred crop` | OpenCV lacks its face data. Clips still render but without face tracking. Run `venv\Scripts\python -m pip install --force-reinstall opencv-python` and try again. |
| `No module named 'cv2'` (or `faster_whisper`) | You are not in the environment. Use `.\clip.ps1`, or activate with `venv\Scripts\activate` and check for `(venv)`. |
| `clip.ps1 ... is not digitally signed` | `Unblock-File C:\clipper\clip.ps1` (needed again after every new download). |
| `A parameter cannot be found that matches parameter ...` | Your `clip.ps1` is old. Install the newest one. |
| `.\clip.ps1 is not recognized` | The file isn't in `C:\clipper` or is named wrong (for example `clip (1).ps1`). Check with `dir C:\clipper`. |
| `Ollama is not running` | Open Ollama from the Start menu, then run again. |
| `Error: pull model manifest: file does not exist` | Typo in the model name. Check spelling and tag. |
| `CUDA out of memory` | Use `-Whisper small`, close GPU-heavy apps. If using `-ThumbVision`, try without it. |
| `The LLM returned no usable clips` | Try `-Min 8`, or `-Llm llama3.1:8b`. |
| Whisper gets many words wrong | Use `-Whisper medium` and set `-Language`. |
| Title box covers a face in the video | Use `-TitleSeconds 3` or `-NoTitle`. |
| Emoji missing in `upload_plan.md` | Open it in Notepad or VS Code (UTF-8). Some viewers don't show emoji. |

If something else breaks, copy the last 10-15 lines from PowerShell and send them to Claude.
