import time
import streamlit as st
import streamlit.components.v1 as components

st.markdown('<style>[data-testid="stSidebar"]{display:none!important;} .block-container{padding:0!important;max-width:100%!important;}</style>', unsafe_allow_html=True)

SPLASH_HTML = """
<!DOCTYPE html>
<html class="light" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<script id="tailwind-config">
  tailwind.config = { darkMode: "class", theme: { extend: {
    "colors": { "primary": "#00264a" },
    "spacing": { "sm":"8px","md":"16px","lg":"24px","xl":"32px" },
  } } };
</script>
<style>
body { font-family:'Public Sans',sans-serif; background-color:#E7EDF3; margin:0; padding:0; overflow:hidden; }
.material-symbols-outlined { font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24; }
@keyframes progress { 0%{width:0%;} 50%{width:70%;} 100%{width:100%;} }
.progress-animate { animation: progress 3s cubic-bezier(0.65,0,0.35,1) forwards; }
@keyframes fadeIn { from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }
.animate-fade-in { animation: fadeIn 0.8s ease-out forwards; }
.loading-delay { animation-delay: 0.3s; }
</style>
</head>
<body class="flex flex-col h-screen items-center justify-center relative">
<div class="absolute top-0 left-0 w-full h-[4px] bg-[#0B3C6B]"></div>
<main class="flex flex-col items-center text-center px-lg max-w-lg">
<div class="w-[120px] h-[120px] rounded-full border border-[#0B3C6B] flex items-center justify-center mb-xl animate-fade-in">
<span class="material-symbols-outlined text-[#0B3C6B] text-[48px]" style="font-variation-settings:'FILL' 0;">account_balance</span>
</div>
<h1 class="text-[24px] font-semibold text-[#0B3C6B] mb-2 animate-fade-in loading-delay">Ministry of Health &amp; Family Welfare</h1>
<p class="text-[12px] font-bold tracking-[0.1em] text-[#B8C6D6] uppercase mb-8 animate-fade-in loading-delay" style="animation-delay:0.5s;">UHC GAP MAPPER — SECURE ACCESS PORTAL</p>
<div class="w-[300px] flex flex-col items-center animate-fade-in" style="animation-delay:0.7s;">
<div class="w-full h-[2px] bg-[#B8C6D6]/30 overflow-hidden mb-2">
<div class="h-full bg-[#0B3C6B] progress-animate"></div>
</div>
<p class="text-[14px] text-[#B8C6D6]" id="status-text">Initializing secure session…</p>
</div>
</main>
<div class="absolute bottom-8 w-full text-center animate-fade-in" style="animation-delay:1s;">
<span class="text-[12px] text-[#B8C6D6]">v2.4.0-GOV | Official Administrative Instance</span>
</div>
<script>
document.addEventListener('DOMContentLoaded', () => {
  const statusText = document.getElementById('status-text');
  const messages = ['Initializing secure session…','Verifying administrative credentials…','Connecting to National Health Database…','Decrypting spatial health data…','Finalizing dashboard environment…'];
  let currentIdx = 0;
  const interval = setInterval(() => {
    currentIdx++;
    if (currentIdx < messages.length) {
      statusText.classList.add('opacity-0');
      setTimeout(() => { statusText.textContent = messages[currentIdx]; statusText.classList.remove('opacity-0'); }, 200);
    } else { clearInterval(interval); statusText.textContent = 'Ready for access'; }
  }, 700);
});
</script>
</body></html>
"""

components.html(SPLASH_HTML, height=700, scrolling=False)

time.sleep(3.6)
st.session_state.splash_shown = True
st.rerun()