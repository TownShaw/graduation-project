<template>
  <div id="main">
    <VideoPost @result-change="getJsonData"/>
    <ResultDisplay :json-data="jsonData"/>
  </div>
</template>

<script>
import VideoPost from "../components/VideoPost.vue"
import ResultDisplay from "../components/ResultDisplay.vue"

export default {
  name: "Main",
  data() {
		return {
      jsonData: {}
    }
  },
  components: {
		VideoPost,
    ResultDisplay
  },
  methods: {
    getJsonData(data) {
      this.jsonData = data
    }
  },
  computed: {
  }
}
</script>

<style>

</style>
