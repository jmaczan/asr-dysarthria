import { AudioManager } from "./components/AudioManager";
import Transcript from "./components/Transcript";
import { useTranscriber } from "./hooks/useTranscriber";

// @ts-ignore
const IS_WEBGPU_AVAILABLE = !!navigator.gpu;

function App() {
    const transcriber = useTranscriber();

    return IS_WEBGPU_AVAILABLE ? (
        <div className='flex justify-center items-center min-h-screen flex-col'>
            <div className='container flex flex-col justify-center items-center'>
                <h1 className='text-5xl font-extrabold tracking-tight text-slate-900 sm:text-7xl text-center'>
                    ASR Dysarthria
                </h1>
                <h2 className='mt-3 mb-5 px-4 text-center text-1xl font-semibold tracking-tight text-slate-900 sm:text-2xl'>
                    Automatic Speech Recognition for dysarthric speech
                </h2>
                <AudioManager transcriber={transcriber} />
                <Transcript transcribedData={transcriber.output} />
            </div>

            <div className='relative bottom-4 text-xs max-w-[40rem]'>
                Made with{" "}
                <a
                    className='underline'
                    href='https://github.com/xenova/transformers.js'
                >
                    🤗 Transformers.js
                </a>
                . Web app based on{" "}
                <a
                    className='underline'
                    href='https://github.com/xenova/whisper-web/tree/experimental-webgpu'
                >
                    xenova/whisper-web on branch experimental-webgpu
                </a>
                . ASR model based on Wav2Vec2 and Patrick von Platen's guide.
                Source code{" "}
                <a
                    className='underline'
                    href='https://github.com/jmaczan/asr-dysarthria'
                >
                    on GitHub repository jmaczan/asr-dysarthria
                </a>
            </div>
        </div>
    ) : (
        <div className='fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] text-white text-2xl font-semibold flex justify-center items-center text-center'>
            Use Chrome browser. WebGPU is not supported by your browser yet
            :&#40;
        </div>
    );
}

export default App;
