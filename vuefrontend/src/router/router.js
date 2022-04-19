// import Vue from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import Main from '../views/Main.vue'

// Vue.use(Router)

export default createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            name: '首页',
            component: Main
        },
        {
            path: '/test',
            name: 'HelloWord Test',
            component: function() {
                return import('../components/HelloWorld.vue')
            }
        }
    ],
    mode: 'history'
})
