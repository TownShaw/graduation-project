<template>
  <div id="main">
    <VideoPost/>
    <!-- <VideoPost @result-change="getJsonData"/> -->
    <!-- <ResultDisplay :json-data="this.jsonData"/> -->
  </div>
</template>

<script>
import VideoPost from "../components/VideoPost.vue"
// import ResultDisplay from "../components/ResultDisplay.vue"

export default {
  name: "Main",
  data() {
		return {
      jsonData: undefined
    }
  },
  components: {
		VideoPost
    // ResultDisplay
  },
  methods: {
    getJsonData(event, data) {
      this.jsonData = data
    }
  },
  computed: {
  }
}
</script>

<style>

</style>
