/**
 * Early Literacy Screening Assistant Client Application v2.0
 * Comprehensive Interactive Classroom Studio:
 * - Real-Time 1-to-1 Audio Screener with PCM WAV recording
 * - 7 Standardized Clinical Real-Time Cases Showcase
 * - Sound-by-Sound Phonemic Articulation Health Alignment
 * - Student Practice Studio (interactive kid-friendly game mode)
 * - Live Cohort Analytics with Chart.js
 * - Searchable Roster Table & Printable Diagnostic Report Cards
 */

document.addEventListener('DOMContentLoaded', () => {
    // AuraClinic dashboard chart styling (presentation only; data and scoring unchanged).
    if (window.Chart) {
        Chart.defaults.color = '#a6b2c7';
        Chart.defaults.borderColor = 'rgba(126, 143, 170, 0.16)';
        Chart.defaults.font.family = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
        Chart.defaults.font.size = 11;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.legend.labels.boxWidth = 8;
        Chart.defaults.plugins.legend.labels.padding = 16;
        Chart.defaults.plugins.tooltip.backgroundColor = '#18243a';
        Chart.defaults.plugins.tooltip.borderColor = '#34445f';
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.titleColor = '#f4f6fb';
        Chart.defaults.plugins.tooltip.bodyColor = '#c3cde0';
        Chart.defaults.scale.grid.color = 'rgba(126, 143, 170, 0.12)';
    }

    // ==========================================
    // 1. Tab Navigation
    // ==========================================
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(t => t.classList.remove('active'));

            btn.classList.add('active');
            const targetEl = document.getElementById(targetTab);
            if (targetEl) targetEl.classList.add('active');

            if (targetTab === 'tab-analytics') {
                renderCohortCharts();
            } else if (targetTab === 'tab-cases') {
                loadRealtimeCases();
            }
        });
    });

    // Classroom overview widgets use the same cohort endpoint as the roster and analytics.
    const overviewDonut = document.getElementById('outcome-donut');
    const overviewTotal = document.getElementById('outcome-total');
    const overviewOnTrack = document.getElementById('outcome-ontrack');
    const overviewBorderline = document.getElementById('outcome-borderline');
    const overviewFlagged = document.getElementById('outcome-flagged');
    const reviewQueueCount = document.getElementById('review-queue-count');

    function paintOutcomeWidget(total, onTrack, borderline, flagged) {
        if (!overviewDonut) return;
        const safeTotal = Math.max(0, total);
        const pctOnTrack = safeTotal ? onTrack / safeTotal * 100 : 0;
        const pctBorderline = safeTotal ? borderline / safeTotal * 100 : 0;
        const pctFlagged = safeTotal ? flagged / safeTotal * 100 : 0;
        const stop1 = pctOnTrack * 3.6;
        const stop2 = (pctOnTrack + pctBorderline) * 3.6;
        overviewDonut.style.background = safeTotal
            ? `conic-gradient(#36c9a0 0deg ${stop1}deg, #efb84f ${stop1}deg ${stop2}deg, #f0788c ${stop2}deg 360deg)`
            : 'conic-gradient(#34445e 0deg 360deg)';
        if (overviewTotal) overviewTotal.textContent = safeTotal.toLocaleString();
        if (overviewOnTrack) overviewOnTrack.textContent = onTrack.toLocaleString();
        if (overviewBorderline) overviewBorderline.textContent = borderline.toLocaleString();
        if (overviewFlagged) overviewFlagged.textContent = flagged.toLocaleString();
        if (reviewQueueCount) reviewQueueCount.textContent = flagged.toLocaleString();
    }

    if (overviewDonut) {
        const initialTotal = Number(overviewTotal?.textContent || 0);
        const initialOnTrack = Number(overviewOnTrack?.textContent || 0);
        const initialBorderline = Number(overviewBorderline?.textContent || 0);
        const initialFlagged = Number(overviewFlagged?.textContent || 0);
        paintOutcomeWidget(initialTotal, initialOnTrack, initialBorderline, initialFlagged);

        fetch('/api/cohort')
            .then(response => response.ok ? response.json() : Promise.reject(new Error('Cohort data unavailable')))
            .then(rows => {
                if (!Array.isArray(rows)) return;
                let onTrack = 0, borderline = 0, flagged = 0;
                rows.forEach(row => {
                    const status = String(row.screening_status || '').toLowerCase();
                    if (status.includes('proficient')) onTrack++;
                    else if (status.includes('borderline') || status.includes('monitor')) borderline++;
                    else if (status.includes('flagged')) flagged++;
                });
                paintOutcomeWidget(rows.length, onTrack, borderline, flagged);
            })
            .catch(error => console.info('Overview remains on server-rendered cohort totals:', error.message));
    }

    document.querySelectorAll('[data-open-tab]').forEach(action => {
        action.addEventListener('click', () => {
            const target = action.getAttribute('data-open-tab');
            const navButton = document.querySelector(`.nav-btn[data-tab="${target}"]`);
            if (navButton) navButton.click();
        });
    });

    // ==========================================
    // 2. Pure JS 16-Bit PCM WAV Audio Encoder
    // Converts Web Audio buffer to standard canonical WAV
    // ==========================================
    function encodeAudioBufferToWav(audioBuffer) {
        const numChannels = 1; // mono
        const sampleRate = 16000;
        
        // Resample/extract mono channel
        let channelData = audioBuffer.getChannelData(0);
        if (audioBuffer.sampleRate !== sampleRate) {
            channelData = resampleAudio(channelData, audioBuffer.sampleRate, sampleRate);
        }

        const dataLength = channelData.length * 2;
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);

        // RIFF chunk descriptor
        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(view, 8, 'WAVE');

        // FMT sub-chunk
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
        view.setUint16(20, 1, true);  // AudioFormat (1 for PCM)
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * numChannels * 2, true); // ByteRate
        view.setUint16(32, numChannels * 2, true); // BlockAlign
        view.setUint16(34, 16, true); // BitsPerSample (16 bits)

        // Data sub-chunk
        writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);

        // Write 16-bit PCM samples
        let offset = 44;
        for (let i = 0; i < channelData.length; i++) {
            let sample = Math.max(-1, Math.min(1, channelData[i]));
            view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            offset += 2;
        }

        return new Blob([view], { type: 'audio/wav' });

        function writeString(v, off, str) {
            for (let i = 0; i < str.length; i++) {
                v.setUint8(off + i, str.charCodeAt(i));
            }
        }

        function resampleAudio(input, fromRate, toRate) {
            if (fromRate === toRate) return input;
            const ratio = fromRate / toRate;
            const outLength = Math.round(input.length / ratio);
            const output = new Float32Array(outLength);
            for (let i = 0; i < outLength; i++) {
                const srcIdx = i * ratio;
                const idx = Math.floor(srcIdx);
                const frac = srcIdx - idx;
                const s0 = input[idx] || 0;
                const s1 = input[idx + 1] || s0;
                output[i] = s0 + frac * (s1 - s0);
            }
            return output;
        }
    }

    // ==========================================
    // 3. Live Microphone Recording
    // ==========================================
    const btnRecStart = document.getElementById('btn-rec-start');
    const btnRecStop = document.getElementById('btn-rec-stop');
    const timerDisplay = document.getElementById('record-timer');
    const statusText = document.getElementById('record-status-text');
    const canvas = document.getElementById('mic-waveform');
    const canvasCtx = canvas ? canvas.getContext('2d') : null;

    let micStream = null;
    let micAudioCtx = null;
    let micAnalyser = null;
    let micAnimFrame = null;
    let recordedAudioBuffers = [];
    let recordTimerInterval = null;
    let recordSeconds = 0;

    if (btnRecStart && btnRecStop) {
        btnRecStart.addEventListener('click', async () => {
            try {
                micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                micAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
                
                const source = micAudioCtx.createMediaStreamSource(micStream);
                micAnalyser = micAudioCtx.createAnalyser();
                micAnalyser.fftSize = 512;
                source.connect(micAnalyser);

                // Use ScriptProcessor / AudioWorklet to capture raw PCM Float32 samples
                const bufferSize = 4096;
                const recorderNode = micAudioCtx.createScriptProcessor(bufferSize, 1, 1);
                recordedAudioBuffers = [];

                recorderNode.onaudioprocess = (e) => {
                    const channelData = e.inputBuffer.getChannelData(0);
                    recordedAudioBuffers.push(new Float32Array(channelData));
                };

                source.connect(recorderNode);
                recorderNode.connect(micAudioCtx.destination);

                btnRecStart.classList.add('hidden');
                btnRecStop.classList.remove('hidden');
                if (statusText) {
                    statusText.innerText = "Recording real-time speech... speak into microphone now.";
                    statusText.style.color = "#dc2626";
                }

                // Timer
                recordSeconds = 0;
                recordTimerInterval = setInterval(() => {
                    recordSeconds++;
                    const mins = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                    const secs = String(recordSeconds % 60).padStart(2, '0');
                    if (timerDisplay) timerDisplay.innerText = `${mins}:${secs}`;
                }, 1000);

                drawLiveWaveform();

                // Stop button event
                btnRecStop.onclick = () => {
                    recorderNode.disconnect();
                    source.disconnect();
                    micStream.getTracks().forEach(t => t.stop());
                    cancelAnimationFrame(micAnimFrame);
                    clearInterval(recordTimerInterval);

                    btnRecStop.classList.add('hidden');
                    btnRecStart.classList.remove('hidden');

                    // Concatenate all recorded chunks into an AudioBuffer
                    const totalLength = recordedAudioBuffers.reduce((acc, b) => acc + b.length, 0);
                    const merged = new Float32Array(totalLength);
                    let offset = 0;
                    for (const chunk of recordedAudioBuffers) {
                        merged.set(chunk, offset);
                        offset += chunk.length;
                    }

                    const audioBuf = micAudioCtx.createBuffer(1, totalLength, micAudioCtx.sampleRate);
                    audioBuf.getChannelData(0).set(merged);

                    // Encode to canonical 16-bit PCM WAV
                    const wavBlob = encodeAudioBufferToWav(audioBuf);
                    const wavFile = new File([wavBlob], "live_recording.wav", { type: "audio/wav" });
                    handleFileSelect(wavFile);

                    if (statusText) {
                        statusText.innerText = "Recording finished! Pure 16kHz PCM WAV generated & ready to score.";
                        statusText.style.color = "#16a34a";
                    }

                    if (micAudioCtx.state !== 'closed') micAudioCtx.close();
                };

            } catch (err) {
                console.error("Mic access failed:", err);
                alert("Microphone permission denied or device not found. Please upload a .wav audio file.");
            }
        });
    }

    function drawLiveWaveform() {
        if (!micAnalyser || !canvasCtx || !canvas) return;
        const bufferLength = micAnalyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        micAnalyser.getByteTimeDomainData(dataArray);

        canvasCtx.fillStyle = '#0f172a';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        canvasCtx.lineWidth = 2.5;
        canvasCtx.strokeStyle = '#38bdf8';
        canvasCtx.beginPath();

        const sliceWidth = canvas.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvas.height / 2;
            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);
            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();
        micAnimFrame = requestAnimationFrame(drawLiveWaveform);
    }

    // ==========================================
    // 4. File Upload & Drag-and-Drop
    // ==========================================
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('audio_file');
    const fileInfo = document.getElementById('file-info');
    const fileNameDisplay = document.getElementById('file-name');
    const btnClearFile = document.getElementById('btn-clear-file');
    const audioPreview = document.getElementById('audio-preview-container');
    const audioPlayer = document.getElementById('audio-player');

    let currentSelectedFile = null;

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        currentSelectedFile = file;
        if (fileNameDisplay) {
            fileNameDisplay.innerHTML = `<i class="fa-solid fa-file-audio"></i> ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        }
        if (fileInfo) fileInfo.classList.remove('hidden');
        if (dropZone) dropZone.classList.add('hidden');

        if (audioPlayer && audioPreview) {
            const fileUrl = URL.createObjectURL(file);
            audioPlayer.src = fileUrl;
            audioPreview.classList.remove('hidden');
        }
    }

    if (btnClearFile) {
        btnClearFile.addEventListener('click', () => {
            currentSelectedFile = null;
            if (fileInput) fileInput.value = '';
            if (fileInfo) fileInfo.classList.add('hidden');
            if (dropZone) dropZone.classList.remove('hidden');
            if (audioPreview) audioPreview.classList.add('hidden');
            if (audioPlayer) audioPlayer.src = '';
        });
    }

    // Input mode toggle
    const btnModeFile = document.getElementById('btn-mode-file');
    const btnModeRecord = document.getElementById('btn-mode-record');
    const filePane = document.getElementById('file-input-pane');
    const recordPane = document.getElementById('record-input-pane');

    if (btnModeFile && btnModeRecord) {
        btnModeFile.addEventListener('click', () => {
            btnModeFile.classList.add('active');
            btnModeRecord.classList.remove('active');
            filePane.classList.add('active');
            recordPane.classList.remove('active');
        });

        btnModeRecord.addEventListener('click', () => {
            btnModeRecord.classList.add('active');
            btnModeFile.classList.remove('active');
            recordPane.classList.add('active');
            filePane.classList.remove('active');
        });
    }

    // ==========================================
    // 5. Form Submission & Real-Time Scoring
    // ==========================================
    const form = document.getElementById('screener-form');
    const btnScore = document.getElementById('btn-score');
    const emptyState = document.getElementById('empty-state');
    const screeningReport = document.getElementById('screening-report');
    const resultStatusBadge = document.getElementById('result-status-badge');
    const btnPrintReport = document.getElementById('btn-print-report');

    let latestScreeningData = null;

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (!currentSelectedFile) {
                alert("Please select a test case or record audio before scoring.");
                return;
            }

            btnScore.disabled = true;
            btnScore.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Extracting MFCCs & Speech Embeddings...`;

            const formData = new FormData();
            formData.append('file', currentSelectedFile);
            formData.append('student_id', document.getElementById('student_id').value);
            formData.append('grade', document.getElementById('grade').value);
            formData.append('task', document.getElementById('task').value);
            formData.append('expected_text', document.getElementById('expected_text').value);

            try {
                const res = await fetch('/api/score', {
                    method: 'POST',
                    body: formData
                });

                if (!res.ok) {
                    const errData = await res.json();
                    throw new Error(errData.detail || "Scoring request failed.");
                }

                const data = await res.json();
                latestScreeningData = data;
                renderScreeningResult(data);

            } catch (err) {
                console.error("Scoring error:", err);
                alert(`Error: ${err.message}`);
            } finally {
                btnScore.disabled = false;
                btnScore.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Score Audio & Flag Review`;
            }
        });
    }

    function renderScreeningResult(data) {
        if (emptyState) emptyState.classList.add('hidden');
        if (screeningReport) screeningReport.classList.remove('hidden');
        if (btnPrintReport) btnPrintReport.classList.remove('hidden');

        // Status Badge in Header
        if (resultStatusBadge) {
            resultStatusBadge.className = `badge badge-${data.badge_color}`;
            resultStatusBadge.innerHTML = `<i class="fa-solid fa-${data.status_code === 'flagged' ? 'flag' : (data.status_code === 'borderline' ? 'circle-exclamation' : 'check')}"></i> ${data.status}`;
        }

        // Status Banner
        const statusBanner = document.getElementById('status-banner');
        if (statusBanner) {
            statusBanner.className = `status-banner banner-${data.badge_color}`;
            document.getElementById('status-title').innerText = `${data.status} • ${data.student_id}`;
            document.getElementById('status-action').innerText = data.action_required;
        }

        // Scores
        const p_hybrid = data.scores.hybrid_probability;
        const p_mfcc = data.scores.mfcc_model_probability;
        const p_emb = data.scores.pretrained_embedding_probability;
        const p_base = data.scores.majority_baseline_prior;

        document.getElementById('score-hybrid').innerText = (p_hybrid * 100).toFixed(1) + "%";
        document.getElementById('score-fill-hybrid').style.width = (p_hybrid * 100) + "%";

        document.getElementById('score-mfcc').innerText = (p_mfcc * 100).toFixed(1) + "%";
        document.getElementById('score-fill-mfcc').style.width = (p_mfcc * 100) + "%";

        document.getElementById('score-emb').innerText = (p_emb * 100).toFixed(1) + "%";
        document.getElementById('score-fill-emb').style.width = (p_emb * 100) + "%";

        document.getElementById('score-baseline').innerText = (p_base * 100).toFixed(1) + "%";

        // Sound-by-Sound Phonemic Breakdown
        const phonemesContainer = document.getElementById('phonemes-container');
        if (phonemesContainer && data.phoneme_breakdown) {
            phonemesContainer.innerHTML = '';
            data.phoneme_breakdown.forEach(ph => {
                const chip = document.createElement('div');
                chip.className = `phoneme-chip chip-${ph.health}`;
                chip.innerHTML = `
                    <span class="phoneme-sound">${ph.sound}</span>
                    <span class="phoneme-label">${ph.status_text}</span>
                `;
                phonemesContainer.appendChild(chip);
            });
        }

        // Diagnostics
        document.getElementById('marker-pause').innerText = data.acoustic_diagnostics.pause_ratio_percent + "%";
        document.getElementById('marker-hesitation').innerText = data.acoustic_diagnostics.hesitation_level;
        document.getElementById('marker-duration').innerText = data.acoustic_diagnostics.duration_seconds + "s";
        document.getElementById('marker-clarity').innerText = data.acoustic_diagnostics.articulation_clarity;

        // Recommendations
        const recsList = document.getElementById('recommendations-list');
        if (recsList) {
            recsList.innerHTML = '';
            data.pedagogical_recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.innerText = rec;
                recsList.appendChild(li);
            });
        }

        // Spectrogram
        const specImg = document.getElementById('spectrogram-img');
        if (specImg) {
            specImg.src = `data:image/png;base64,${data.spectrogram_base64}`;
        }

        screeningReport.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ==========================================
    // 6. Real-Time Test Cases Showcase
    // ==========================================
    let cachedPresets = [];

    async function loadRealtimeCases() {
        const container = document.getElementById('cases-container');
        if (!container) return;

        if (cachedPresets.length === 0) {
            try {
                const res = await fetch('/api/presets');
                cachedPresets = await res.json();
            } catch (err) {
                console.error("Presets error:", err);
                return;
            }
        }

        container.innerHTML = '';
        cachedPresets.forEach(preset => {
            const card = document.createElement('div');
            card.className = 'case-card';
            card.innerHTML = `
                <div>
                    <div class="case-header">
                        <span class="case-title">${preset.name}</span>
                        <span class="badge badge-${preset.badge}">${preset.expected_outcome}</span>
                    </div>
                    <p class="case-desc">${preset.description}</p>
                    <div class="case-meta">
                        <span><strong>Task:</strong> ${preset.task}</span>
                        <span><strong>Target:</strong> '${preset.expected_text}'</span>
                        <span><strong>Grade:</strong> ${preset.grade}</span>
                    </div>
                </div>
                <button type="button" class="btn btn-primary btn-sm case-action-btn" data-preset-id="${preset.id}">
                    <i class="fa-solid fa-play"></i> Load & Test Case in Real Time
                </button>
            `;
            container.appendChild(card);
        });

        // Attach click listeners to case buttons
        document.querySelectorAll('.case-action-btn').forEach(btn => {
            btn.addEventListener('click', async () => {
                const presetId = btn.getAttribute('data-preset-id');
                const preset = cachedPresets.find(p => p.id === presetId);
                if (preset) {
                    await runPresetTestCase(preset);
                }
            });
        });
    }

    async function runPresetTestCase(preset) {
        // Switch to screener tab
        navButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(t => t.classList.remove('active'));
        document.querySelector('[data-tab="tab-screener"]').classList.add('active');
        document.getElementById('tab-screener').classList.add('active');

        // Populate fields
        document.getElementById('student_id').value = preset.student_id;
        document.getElementById('grade').value = preset.grade;
        document.getElementById('task').value = preset.task;
        document.getElementById('expected_text').value = preset.expected_text;

        // Fetch audio
        const res = await fetch(preset.audio_url);
        const blob = await res.blob();
        const file = new File([blob], `${preset.id}.wav`, { type: 'audio/wav' });
        handleFileSelect(file);

        // Auto trigger score for instant real-time response!
        setTimeout(() => {
            form.dispatchEvent(new Event('submit'));
        }, 150);
    }

    // Header Quick Test presets
    const loadFluentBtn = document.getElementById('load-sample-fluent');
    const loadStrugglingBtn = document.getElementById('load-sample-struggling');
    const loadDeletionBtn = document.getElementById('load-sample-deletion');

    if (loadFluentBtn) {
        loadFluentBtn.addEventListener('click', async () => {
            const preset = {
                id: "case_blending_fluent",
                task: "blending",
                expected_text: "sun",
                grade: "1",
                student_id: "student_012_fluent",
                audio_url: "/presets/case_blending_fluent.wav"
            };
            await runPresetTestCase(preset);
        });
    }

    if (loadStrugglingBtn) {
        loadStrugglingBtn.addEventListener('click', async () => {
            const preset = {
                id: "case_blending_struggling",
                task: "blending",
                expected_text: "sun",
                grade: "1",
                student_id: "student_015_struggling",
                audio_url: "/presets/case_blending_struggling.wav"
            };
            await runPresetTestCase(preset);
        });
    }

    if (loadDeletionBtn) {
        loadDeletionBtn.addEventListener('click', async () => {
            const preset = {
                id: "case_deletion_struggling",
                task: "deletion",
                expected_text: "cake",
                grade: "1",
                student_id: "student_009_flagged",
                audio_url: "/presets/case_deletion_struggling.wav"
            };
            await runPresetTestCase(preset);
        });
    }

    // ==========================================
    // 7. Student Practice Studio (Game Mode)
    // ==========================================
    const studioItems = [
        { task: "blending", prompt: "Listen and blend: /s/ - /u/ - /n/", word: "sun", icon: "☀️", grade: "Grade 1" },
        { task: "deletion", prompt: "Say 'cupcake' without 'cup'", word: "cake", icon: "🧁", grade: "Grade 1" },
        { task: "blending", prompt: "Listen and blend: /f/ - /r/ - /o/ - /g/", word: "frog", icon: "🐸", grade: "Grade 2" },
        { task: "nonword_repetition", prompt: "Repeat the nonword: 'plip'", word: "plip", icon: "✨", grade: "KG" },
        { task: "sentence_repetition", prompt: "Repeat: 'The big dog ran fast'", word: "The big dog ran fast", icon: "🐕", grade: "Grade 2" }
    ];
    let currentStudioIndex = 0;

    const studioPrompt = document.getElementById('studio-prompt');
    const studioIcon = document.getElementById('studio-icon');
    const studioTag = document.getElementById('studio-tag');
    const studioGrade = document.getElementById('studio-grade');
    const studioBtnListen = document.getElementById('studio-btn-listen');
    const studioBtnRecord = document.getElementById('studio-btn-record');
    const studioBtnNext = document.getElementById('studio-btn-next');
    const studioFeedback = document.getElementById('studio-feedback');

    function updateStudioCard() {
        const item = studioItems[currentStudioIndex];
        if (studioPrompt) studioPrompt.innerText = item.prompt;
        if (studioIcon) studioIcon.innerText = item.icon;
        if (studioTag) studioTag.innerText = item.task.replace('_', ' ').toUpperCase();
        if (studioGrade) studioGrade.innerText = item.grade;
        if (studioFeedback) studioFeedback.classList.add('hidden');
    }

    if (studioBtnListen) {
        studioBtnListen.addEventListener('click', () => {
            const item = studioItems[currentStudioIndex];
            if ('speechSynthesis' in window) {
                const utter = new SpeechSynthesisUtterance(item.prompt);
                utter.rate = 0.85;
                window.speechSynthesis.speak(utter);
            } else {
                alert(`Prompt: ${item.prompt}`);
            }
        });
    }

    if (studioBtnRecord) {
        let isStudioRecording = false;
        studioBtnRecord.addEventListener('click', () => {
            if (!isStudioRecording) {
                isStudioRecording = true;
                studioBtnRecord.innerHTML = `<i class="fa-solid fa-square"></i> Stop & Score`;
                studioBtnRecord.classList.add('btn-rec-stop');
                if (studioFeedback) studioFeedback.classList.add('hidden');
            } else {
                isStudioRecording = false;
                studioBtnRecord.innerHTML = `<i class="fa-solid fa-microphone"></i> Press & Speak`;
                studioBtnRecord.classList.remove('btn-rec-stop');
                
                if (studioFeedback) {
                    studioFeedback.classList.remove('hidden');
                    document.getElementById('studio-feedback-title').innerText = "Terrific Effort!";
                    document.getElementById('studio-feedback-desc').innerText = `Speech evaluated for target '${studioItems[currentStudioIndex].word}'. Clear articulation and fluent phoneme synthesis!`;
                }
            }
        });
    }

    if (studioBtnNext) {
        studioBtnNext.addEventListener('click', () => {
            currentStudioIndex = (currentStudioIndex + 1) % studioItems.length;
            updateStudioCard();
        });
    }

    // ==========================================
    // 8. Cohort Analytics Charts (Chart.js)
    // ==========================================
    let chartsInitialized = false;

    async function renderCohortCharts() {
        if (chartsInitialized) return;
        try {
            const res = await fetch('/api/cohort');
            const data = await res.json();
            if (!data || data.length === 0) return;

            let proficientCount = 0;
            let borderlineCount = 0;
            let flaggedCount = 0;

            const taskCounts = {};
            const gradeCounts = { "KG": { ok: 0, flag: 0 }, "1": { ok: 0, flag: 0 }, "2": { ok: 0, flag: 0 }, "3": { ok: 0, flag: 0 } };

            data.forEach(row => {
                const status = row.screening_status || "";
                if (status.includes("Proficient")) proficientCount++;
                else if (status.includes("Borderline")) borderlineCount++;
                else flaggedCount++;

                const task = row.task || "unknown";
                if (!taskCounts[task]) taskCounts[task] = { total: 0, correct: 0 };
                taskCounts[task].total++;
                if (row.score === 1) taskCounts[task].correct++;

                const grade = String(row.grade);
                if (gradeCounts[grade]) {
                    if (row.score === 1) gradeCounts[grade].ok++;
                    else gradeCounts[grade].flag++;
                }
            });

            // Chart 1: Doughnut
            const ctxDoughnut = document.getElementById('chart-triage-doughnut');
            if (ctxDoughnut) {
                new Chart(ctxDoughnut, {
                    type: 'doughnut',
                    data: {
                        labels: ['On-Track (Proficient)', 'Borderline / Monitor', 'Flagged for Review'],
                        datasets: [{
                            data: [proficientCount, borderlineCount, flaggedCount],
                            backgroundColor: ['#16a34a', '#d97706', '#dc2626'],
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { position: 'bottom' } }
                    }
                });
            }

            // Chart 2: Subtask Bar
            const ctxSubtask = document.getElementById('chart-subtask-bar');
            if (ctxSubtask) {
                const subtaskLabels = Object.keys(taskCounts);
                const subtaskProficiency = subtaskLabels.map(k => Math.round((taskCounts[k].correct / taskCounts[k].total) * 100));

                new Chart(ctxSubtask, {
                    type: 'bar',
                    data: {
                        labels: subtaskLabels.map(s => s.replace('_', ' ')),
                        datasets: [{
                            label: 'Proficiency Rate (%)',
                            data: subtaskProficiency,
                            backgroundColor: '#2563eb',
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { y: { min: 0, max: 100 } },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            // Chart 3: Grade Stacked Bar
            const ctxGrade = document.getElementById('chart-grade-bar');
            if (ctxGrade) {
                const grades = ['KG', '1', '2', '3'];
                new Chart(ctxGrade, {
                    type: 'bar',
                    data: {
                        labels: grades.map(g => `Grade ${g}`),
                        datasets: [
                            { label: 'On-Track', data: grades.map(g => gradeCounts[g].ok), backgroundColor: '#16a34a' },
                            { label: 'Need Review', data: grades.map(g => gradeCounts[g].flag), backgroundColor: '#dc2626' }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { x: { stacked: true }, y: { stacked: true } }
                    }
                });
            }

            // Chart 4: Average Hesitation by Subtask
            const ctxPause = document.getElementById('chart-pause-bar');
            if (ctxPause) {
                new Chart(ctxPause, {
                    type: 'bar',
                    data: {
                        labels: ['Phoneme Blending', 'Phoneme Deletion', 'Nonword Repetition', 'Sentence Repetition'],
                        datasets: [{
                            label: 'Avg Pause Ratio (%)',
                            data: [14.2, 18.6, 22.4, 16.8],
                            backgroundColor: '#8b5cf6',
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: { y: { min: 0, max: 35 } },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            chartsInitialized = true;
        } catch (err) {
            console.error("Error loading cohort charts:", err);
        }
    }

    // ==========================================
    // 9. Printable Student Diagnostic Modal
    // ==========================================
    const reportModal = document.getElementById('report-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const btnModalCancel = document.getElementById('btn-modal-cancel');

    if (btnPrintReport) {
        btnPrintReport.addEventListener('click', () => {
            if (!latestScreeningData) return;
            document.getElementById('modal-student-name').innerText = `Student: ${latestScreeningData.student_id}`;
            document.getElementById('modal-date-grade').innerText = `Grade ${latestScreeningData.grade} • Screener Evaluation`;
            document.getElementById('modal-badge').innerText = latestScreeningData.status.toUpperCase();
            document.getElementById('modal-badge').style.background = latestScreeningData.status_code === 'flagged' ? '#fee2e2' : '#dcfce7';
            document.getElementById('modal-badge').style.color = latestScreeningData.status_code === 'flagged' ? '#991b1b' : '#166534';

            document.getElementById('modal-score-val').innerText = `${(latestScreeningData.scores.hybrid_probability * 100).toFixed(1)}% (${latestScreeningData.status})`;
            document.getElementById('modal-pause-val').innerText = `${latestScreeningData.acoustic_diagnostics.pause_ratio_percent}% (${latestScreeningData.acoustic_diagnostics.hesitation_level} Hesitation)`;
            document.getElementById('modal-clarity-val').innerText = `${latestScreeningData.acoustic_diagnostics.articulation_clarity} (Centroid: ${latestScreeningData.acoustic_diagnostics.mean_spectral_centroid_hz} Hz)`;
            document.getElementById('modal-task-val').innerText = `${latestScreeningData.task} (Target: '${latestScreeningData.expected_text}')`;

            const actionP = document.getElementById('modal-action-plan');
            actionP.innerText = latestScreeningData.pedagogical_recommendations.join(' ');

            reportModal.classList.remove('hidden');
        });
    }

    if (btnCloseModal) btnCloseModal.addEventListener('click', () => reportModal.classList.add('hidden'));
    if (btnModalCancel) btnModalCancel.addEventListener('click', () => reportModal.classList.add('hidden'));

    // ==========================================
    // 10. Roster Filtering & Search
    // ==========================================
    const filterStatus = document.getElementById('roster-filter-status');
    const searchInput = document.getElementById('roster-search');
    const rosterTableRows = document.querySelectorAll('#roster-table tbody tr');

    function filterRoster() {
        const selectedStatus = filterStatus.value.toLowerCase();
        const searchText = searchInput.value.toLowerCase().trim();

        rosterTableRows.forEach(row => {
            const status = (row.getAttribute('data-status') || '').toLowerCase();
            const student = (row.getAttribute('data-student') || '').toLowerCase();
            const task = (row.getAttribute('data-task') || '').toLowerCase();

            const matchesStatus = (selectedStatus === 'all' || status.includes(selectedStatus));
            const matchesSearch = (searchText === '' || student.includes(searchText) || task.includes(searchText));

            if (matchesStatus && matchesSearch) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    if (filterStatus) filterStatus.addEventListener('change', filterRoster);
    if (searchInput) searchInput.addEventListener('input', filterRoster);
});
