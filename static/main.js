import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

let prompt = '@@ПЕРВЫЙ@@винда или линукс или макос?@@ВТОРОЙ@@'

const clock = new THREE.Clock()

const scene = new THREE.Scene()
scene.background = new THREE.Color(0xdddddd)

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 5000)
camera.position.set(-29.2, 18.3, 6.3) 
camera.rotation.set(-1, -1.1, -1)

const light = new THREE.AmbientLight(0x404040, 100)
scene.add(light)

const light1 = new THREE.PointLight(0xffffff, 2)
light1.position.set(2.5, 2.5, 2.5)
scene.add(light1)

const light2 = new THREE.PointLight(0xffffff, 2)
light2.position.set(-2.5, 2.5, 2.5)
scene.add(light2)

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(window.devicePixelRatio)
document.body.appendChild(renderer.domElement)

// const controls = new OrbitControls(camera, renderer.domElement)
// controls.addEventListener('change', renderer)

const subtitle = document.getElementById('subtitle')

let irinaTalking = false
let pavelTalking = false

const loader = new GLTFLoader()

loader.load('static/models/scene.glb', (gltf) => {
    const model = gltf.scene

    scene.add(model)
    animate()
})

let pavelMixer
let pavelAnimation
let pavelModel
loader.load('static/models/pavel.glb', (gltf) => {
    pavelModel = gltf.scene

    pavelModel.position.z = 10
    pavelModel.rotation.y = 60 * Math.PI / 180

    pavelAnimation = gltf.animations[0]
    pavelMixer = new THREE.AnimationMixer(pavelModel)

    scene.add(pavelModel)
    animate()
})

let irinaMixer
let irinaAnimation
let irinaModel
loader.load('static/models/irina.glb', (gltf) => {
    irinaModel = gltf.scene

    irinaModel.position.z = -10
    irinaModel.position.y = 1.5
    irinaModel.rotation.y = -40 * Math.PI / 180
    
    irinaAnimation = gltf.animations[0]
    irinaMixer = new THREE.AnimationMixer(irinaModel)
    
    scene.add(irinaModel)
    animate()
})

function animate() {
    requestAnimationFrame(animate)

    const delta = clock.getDelta()

    if(pavelMixer) {
        if(pavelTalking) {
            pavelMixer.clipAction(pavelAnimation).play()
        } else {
            pavelMixer.clipAction(pavelAnimation).stop()
        }

        pavelMixer.update(delta)
    }

    if(irinaMixer) {
        if(irinaTalking) {
            irinaMixer.clipAction(irinaAnimation).play()
        } else {
            irinaMixer.clipAction(irinaAnimation).stop()
        }

        irinaMixer.update(delta)
    }

    renderer.render(scene, camera)
}

let irinaVoice
let pavelVoice
window.speechSynthesis.onvoiceschanged = () => {
    const voices = speechSynthesis.getVoices()
    console.log(voices)

    irinaVoice = voices[3]
    pavelVoice = voices[4]
}

async function fetchReplyAndSpeak(speaker) {
    let result
    result = await fetch('/api/generate?&p=' + prompt + '&speaker=' + speaker)
    
    const data = await result.text()
    console.log('Reply: ' + data)
    
    const speech = new SpeechSynthesisUtterance()
    
    speech.lang = 'ru'
    speech.text = data

    if(speaker == 'Irina') {
        if(irinaModel) camera.lookAt(irinaModel.position)
        speech.voice = irinaVoice
    } else if(speaker == 'Pavel') {
        if(pavelModel) camera.lookAt(pavelModel.position)
        speech.voice = pavelVoice
    }

    window.speechSynthesis.speak(speech)

    if(speaker == 'Irina') {
        speech.onstart = () => {
            irinaTalking = true
        }
        speech.onend = () => {
            irinaTalking = false
        }
    } else if(speaker == 'Pavel') {
        speech.onstart = () => {
            pavelTalking = true
        }
        speech.onend = () => {
            pavelTalking = false
        }
    }

    return data
}

while(true) {
    prompt = await fetchReplyAndSpeak('Irina')
    subtitle.innerText = prompt
    subtitle.style.color = '#ff00d2'

    prompt = `@@ПЕРВЫЙ@@${prompt}@@ВТОРОЙ@@`

    prompt = await fetchReplyAndSpeak('Pavel')
    subtitle.innerText = prompt
    subtitle.style.color = 'blue'

    prompt = `@@ПЕРВЫЙ@@${prompt}@@ВТОРОЙ@@`
}
