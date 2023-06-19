import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

let prompt = '@@ПЕРВЫЙ@@ты тупой@@ВТОРОЙ@@'

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

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.setPixelRatio(window.devicePixelRatio)
document.body.appendChild(renderer.domElement)

// const controls = new OrbitControls(camera, renderer.domElement)
// controls.addEventListener('change', renderer)

const subtitle = document.getElementById('subtitle')

let womanTalking = false
let manTalking = false

const loader = new GLTFLoader()

loader.load('static/models/scene.glb', (gltf) => {
    const model = gltf.scene

    scene.add(model)
    animate()
})

let manMixer
let manAnimation
let manModel
loader.load('static/models/man.glb', (gltf) => {
    manModel = gltf.scene

    manModel.position.z = 10
    manModel.rotation.y = 60 * Math.PI / 180

    manAnimation = gltf.animations[0]
    manMixer = new THREE.AnimationMixer(manModel)

    scene.add(manModel)
    animate()
})

let womanMixer
let womanAnimation
let womanModel
loader.load('static/models/woman.glb', (gltf) => {
    womanModel = gltf.scene

    womanModel.position.z = -10
    womanModel.position.y = 1.5
    womanModel.rotation.y = -40 * Math.PI / 180
    
    womanAnimation = gltf.animations[0]
    womanMixer = new THREE.AnimationMixer(womanModel)
    
    scene.add(womanModel)
    animate()
})

function animate() {
    requestAnimationFrame(animate)

    const delta = clock.getDelta()

    if(manMixer) {
        if(manTalking) {
            manMixer.clipAction(manAnimation).play()
        } else {
            manMixer.clipAction(manAnimation).stop()
        }

        manMixer.update(delta)
    }

    if(womanMixer) {
        if(womanTalking) {
            womanMixer.clipAction(womanAnimation).play()
        } else {
            womanMixer.clipAction(womanAnimation).stop()
        }

        womanMixer.update(delta)
    }

    renderer.render(scene, camera)
}

async function fetchReplyAndSpeak(prompt, speaker) {
    const result = await fetch('/api/generate?p=' + prompt + '&speaker=' + speaker)

    const data = await result.json()
    console.log('Reply: ' + data.reply)
    
    if(speaker == 'woman') {
        if(womanModel) camera.lookAt(womanModel.position)

        womanTalking = true
        setTimeout(() => {
            womanTalking = false
        }, data.duration * 1000)
    } else if(speaker == 'man') {
        if(manModel) camera.lookAt(manModel.position)

        manTalking = true
        setTimeout(() => {
            manTalking = false
        }, data.duration * 1000)
    }

    return data.reply
}

while(true) {
    prompt = await fetchReplyAndSpeak(prompt, 'woman')
    subtitle.innerText = prompt
    subtitle.style.color = '#ff00d2'

    prompt = `@@ПЕРВЫЙ@@${prompt}@@ВТОРОЙ@@`

    prompt = await fetchReplyAndSpeak(prompt, 'man')
    subtitle.innerText = prompt
    subtitle.style.color = 'blue'

    prompt = `@@ПЕРВЫЙ@@${prompt}@@ВТОРОЙ@@`
}
